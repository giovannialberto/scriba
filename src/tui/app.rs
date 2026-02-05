use crate::core::{
    CompressionSettings, LocalModelSize, RecordOptions, ScribaConfig, TranscriptionMode,
    WorkflowManager, record_audio,
};
use crate::database::{Database, Recording, RecordingStats};
use crate::utils::generate_recording_name;
use anyhow::Result;
use crossterm::{
    event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyEventKind},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Alignment, Constraint, Direction, Layout},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Cell, Clear, Paragraph, Row, Table, TableState, Wrap},
    Frame, Terminal,
};
use std::io;
use tokio::sync::mpsc;

const ASCII_ART: &str = r#" ███████  ██████ ██████  ██ ██████   █████
██      ██      ██   ██ ██ ██   ██ ██   ██
███████ ██      ██████  ██ ██████  ███████
     ██ ██      ██   ██ ██ ██   ██ ██   ██
███████  ██████ ██   ██ ██ ██████  ██   ██"#;
use anyhow::Context;
use dirs::home_dir;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::Duration;
use tokio::process::Command as TokioCommand;

pub struct Dashboard {
    db: Database,
    recordings: Vec<Recording>,
    table_state: TableState,
    current_page: usize,
    page_size: usize,
    stats: Option<RecordingStats>,
    show_help: bool,
    current_view: DashboardView,
    search_mode: bool,
    search_query: String,
    show_message: bool,
    message: String,
    show_transcript: bool,
    transcript_content: String,
    transcript_scroll_offset: usize,
    show_delete_confirm: bool,
    delete_candidate: Option<Recording>,
    current_playback_pid: Option<u32>,
    playback_finished_rx: Option<mpsc::Receiver<()>>, // Channel to receive playback completion
    last_transcribe_warning: Option<usize>, // Track which recording showed overwrite warning
    progress_animation: Option<String>,     // Base message for progress animation
    progress_frame: usize,                  // Animation frame counter
    transcription_task: Option<tokio::task::JoinHandle<Result<(), anyhow::Error>>>, // Background transcription task
    recording_task: Option<tokio::task::JoinHandle<Result<String, anyhow::Error>>>, // Background recording task (returns recording name)
    recording_mode: Option<RecordingMode>, // Track if we should transcribe after recording
    recording_stop_tx: Option<mpsc::Sender<()>>, // Channel to stop recording
    recording_level_rx: Option<mpsc::Receiver<f32>>, // Channel to receive volume levels
    current_volume_level: f32,             // Current recording volume for display
    config: ScribaConfig,                  // App configuration
    settings_selection: usize,             // Current setting selection
    editing_api_key: bool,                 // Whether we're editing API key
    api_key_input: String,                 // API key input buffer
    return_to_view: Option<DashboardView>, // View to return to after message dismissal
    // File import dialog state
    show_file_dialog: bool,
    file_path_input: String,
    file_name_input: String,
    file_dialog_stage: FileDialogStage, // Current stage of file import process
    is_retranscribing: bool,            // Track if current transcription is a re-transcription
    is_importing: bool,                 // Track if current task is import+transcribe
}

#[derive(Debug, PartialEq)]
enum DashboardView {
    Main,
    Help,
    Settings,
}

#[derive(Debug, PartialEq)]
enum FileDialogStage {
    FilePath, // Asking for file path
    FileName, // Asking for display name (optional)
}

#[derive(Debug, Clone)]
enum RecordingMode {
    RecordAndTranscribe,
}

#[derive(Debug)]
enum DashboardAction {
    Continue,
    Quit,
    RecordAndTranscribe,
    AddExternalFile,
    TranscribeSelected,
}

impl Dashboard {
    pub fn new() -> Result<Self> {
        let db = Database::new()?;
        let mut table_state = TableState::default();
        table_state.select(Some(0));
        let config = ScribaConfig::load()?;

        Ok(Self {
            db,
            recordings: Vec::new(),
            table_state,
            current_page: 0,
            page_size: 50, // Show more recordings per page
            stats: None,
            show_help: false,
            current_view: DashboardView::Main,
            search_mode: false,
            search_query: String::new(),
            show_message: false,
            message: String::new(),
            show_transcript: false,
            transcript_content: String::new(),
            transcript_scroll_offset: 0,
            show_delete_confirm: false,
            delete_candidate: None,
            current_playback_pid: None,
            playback_finished_rx: None,
            last_transcribe_warning: None,
            progress_animation: None,
            progress_frame: 0,
            transcription_task: None,
            recording_task: None,
            recording_mode: None,
            recording_stop_tx: None,
            recording_level_rx: None,
            current_volume_level: 0.0,
            config,
            settings_selection: 0,
            editing_api_key: false,
            api_key_input: String::new(),
            return_to_view: None,
            // File import dialog state
            show_file_dialog: false,
            file_path_input: String::new(),
            file_name_input: String::new(),
            file_dialog_stage: FileDialogStage::FilePath,
            is_retranscribing: false,
            is_importing: false,
        })
    }

    pub async fn run(&mut self) -> Result<()> {
        // Setup terminal
        enable_raw_mode()?;
        let mut stdout = io::stdout();
        execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
        let backend = CrosstermBackend::new(stdout);
        let mut terminal = Terminal::new(backend)?;

        // Load initial data
        self.load_recordings()?;
        self.load_stats()?;

        let result = self.run_app(&mut terminal).await;

        // Restore terminal
        disable_raw_mode()?;
        execute!(
            terminal.backend_mut(),
            LeaveAlternateScreen,
            DisableMouseCapture
        )?;
        terminal.show_cursor()?;

        result
    }

    async fn run_app<B: ratatui::backend::Backend>(
        &mut self,
        terminal: &mut Terminal<B>,
    ) -> Result<()> {
        loop {
            // Check if recording task completed
            if let Some(task) = &mut self.recording_task {
                if task.is_finished() {
                    let completed_task = self.recording_task.take().unwrap();
                    let recording_mode = self.recording_mode.take();

                    // Clean up channels
                    self.recording_stop_tx = None;
                    self.recording_level_rx = None;
                    self.current_volume_level = 0.0;

                    match completed_task.await {
                        Ok(Ok(recording_name)) => {
                            // Recording completed successfully
                            if let Some(RecordingMode::RecordAndTranscribe) = recording_mode {
                                // Start transcription phase (local, no API key)
                                // Update progress message for transcription phase
                                let model_info = match &self.config.transcription {
                                    TranscriptionMode::Local { model_size } => {
                                        format!("Local ({})", model_size)
                                    }
                                    TranscriptionMode::Api { .. } => {
                                        "OpenAI API".to_string()
                                    }
                                };
                                self.progress_animation =
                                    Some(format!("📝 Transcribing with {}", model_info));
                                self.progress_frame = 0;

                                // Start transcription
                                // Start auto-transcription using unified workflow
                                let transcription_mode = self.config.transcription.clone();
                                let recording_name_clone = recording_name.clone();

                                self.transcription_task = Some(tokio::spawn(async move {
                                    let mut workflow = WorkflowManager::new().unwrap();
                                    workflow
                                        .retranscribe_recording_silent(
                                            &recording_name_clone,
                                            transcription_mode,
                                        )
                                        .await
                                }));
                            } else {
                                // Recording only mode - complete
                                self.stop_progress_animation();
                                self.message = "✅ Recording complete!".to_string();
                                self.show_message = true;
                                // Reload data to show new recording
                                let _ = self.load_recordings();
                                let _ = self.load_stats();
                            }
                        }
                        Ok(Err(err)) => {
                            self.stop_progress_animation();
                            self.message = format!("❌ Recording failed: {}", err);
                            self.show_message = true;
                        }
                        Err(_) => {
                            self.stop_progress_animation();
                            self.message = "❌ Recording task failed".to_string();
                            self.show_message = true;
                        }
                    }
                }
            }

            // Check if transcription task completed
            if let Some(task) = &mut self.transcription_task {
                if task.is_finished() {
                    let completed_task = self.transcription_task.take().unwrap();
                    match completed_task.await {
                        Ok(Ok(())) => {
                            self.stop_progress_animation();
                            self.message = if self.is_retranscribing {
                                "✅ Re-transcription complete!".to_string()
                            } else if self.is_importing {
                                "✅ Import and transcription complete!".to_string()
                            } else {
                                "✅ Recording and transcription complete!".to_string()
                            };
                            self.show_message = true;
                            self.is_retranscribing = false;
                            self.is_importing = false;
                            // Reload data to show updated transcript
                            let _ = self.load_recordings();
                            let _ = self.load_stats();
                        }
                        Ok(Err(err)) => {
                            self.stop_progress_animation();
                            self.message = format!("❌ Transcription failed: {}", err);
                            self.show_message = true;
                            self.is_retranscribing = false;
                        }
                        Err(_) => {
                            self.stop_progress_animation();
                            self.message = "❌ Transcription task failed".to_string();
                            self.show_message = true;
                            self.is_retranscribing = false;
                        }
                    }
                }
            }

            // Receive volume levels from recording
            if let Some(level_rx) = &mut self.recording_level_rx {
                if let Ok(level) = level_rx.try_recv() {
                    self.current_volume_level = level;
                }
            }

            // Check for playback completion
            if let Some(finished_rx) = &mut self.playback_finished_rx {
                if finished_rx.try_recv().is_ok() {
                    self.current_playback_pid = None;
                    self.playback_finished_rx = None;
                }
            }

            // Update progress animation if active
            if self.progress_animation.is_some() {
                self.update_progress_message();
            }

            terminal.draw(|f| self.ui(f))?;

            if event::poll(Duration::from_millis(100))? {
                if let Event::Key(key) = event::read()? {
                    if key.kind == KeyEventKind::Press {
                        match self.handle_key_event(key.code).await {
                            Ok(DashboardAction::Continue) => continue,
                            Ok(DashboardAction::Quit) => break,
                            Ok(action) => {
                                // Handle other actions (like launching recording, etc.)
                                self.handle_dashboard_action(action).await?;
                            }
                            Err(e) => return Err(e),
                        }
                    }
                }
            }
        }
        Ok(())
    }

    async fn handle_key_event(&mut self, key_code: KeyCode) -> Result<DashboardAction> {
        // If audio is playing and ESC is pressed, stop it immediately
        if matches!(key_code, KeyCode::Esc) {
            // Check if we're in audio playback mode (either with PID or with playback message)
            let is_playing_audio = self.current_playback_pid.is_some()
                || (self.show_message && self.message.contains("Playing:"));

            if is_playing_audio {
                // Try both methods to ensure reliable stopping
                if let Some(pid) = self.current_playback_pid {
                    self.stop_audio_playback(pid)?;
                }
                // Also use emergency stop as a fallback (in case PID method fails)
                self.emergency_stop_all_audio_players()?;

                // Clear playback state
                self.current_playback_pid = None;
                self.playback_finished_rx = None;
                self.show_message = false;
                self.message.clear();

                // Audio playback stops immediately - return to dashboard
                return Ok(DashboardAction::Continue);
            }
        }

        // If recording is active and Escape is pressed, stop recording
        if self.recording_task.is_some() && matches!(key_code, KeyCode::Esc) {
            // Send stop signal to recording task
            if let Some(stop_tx) = self.recording_stop_tx.take() {
                let _ = stop_tx.send(()).await;
            }
            // The recording task will handle cleanup and completion
            return Ok(DashboardAction::Continue);
        }

        if self.show_file_dialog {
            return self.handle_file_dialog_keys(key_code).await;
        }

        if self.show_message {
            // Special-case: allow confirming re-transcribe overwrite with T while message is visible
            if matches!(key_code, KeyCode::Char('t') | KeyCode::Char('T'))
                && self.last_transcribe_warning.is_some()
                && self.progress_animation.is_none()
            {
                // Dismiss the warning and trigger the action
                self.show_message = false;
                self.message.clear();
                return Ok(DashboardAction::TranscribeSelected);
            }

            // Don't close message if progress animation is active
            if self.progress_animation.is_none() && matches!(key_code, KeyCode::Esc) {
                // Only Esc key closes the message popup (consistent behavior)
                self.show_message = false;
                self.message.clear();

                // Return to the previous view if one was set
                if let Some(return_view) = self.return_to_view.take() {
                    self.current_view = return_view;
                }
            }
            return Ok(DashboardAction::Continue);
        }

        if self.search_mode {
            return self.handle_search_input(key_code).await;
        }

        if self.show_help {
            self.show_help = false;
            self.current_view = DashboardView::Main;
            return Ok(DashboardAction::Continue);
        }

        if self.current_view == DashboardView::Settings {
            return self.handle_settings_keys(key_code).await;
        }

        if self.show_transcript {
            return self.handle_transcript_keys(key_code).await;
        }

        if self.show_delete_confirm {
            return self.handle_delete_confirmation(key_code).await;
        }

        match key_code {
            KeyCode::Char('q') | KeyCode::Esc => return Ok(DashboardAction::Quit),
            KeyCode::Char('h') | KeyCode::Char('H') | KeyCode::F(1) => {
                self.show_help = true;
                self.current_view = DashboardView::Help;
            }
            KeyCode::Char('s') | KeyCode::Char('S') => {
                self.current_view = DashboardView::Settings;
                self.settings_selection = 0;
            }
            KeyCode::Char('r') | KeyCode::Char('R') => {
                return Ok(DashboardAction::RecordAndTranscribe);
            }
            KeyCode::Char('a') | KeyCode::Char('A') => {
                return Ok(DashboardAction::AddExternalFile);
            }
            KeyCode::Char('t') | KeyCode::Char('T') => {
                return Ok(DashboardAction::TranscribeSelected);
            }
            KeyCode::Char('/') => {
                self.search_mode = true;
                self.search_query.clear();
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
                    self.message = format!("❌ Failed to load transcript: {}", e);
                    self.show_message = true;
                }
            },
            KeyCode::Char('d') => {
                self.show_delete_confirmation();
            }
            KeyCode::Delete => {
                self.show_delete_confirmation();
            }
            KeyCode::Char('p') | KeyCode::Char('P') => match self.play_selected_recording().await {
                Ok(()) => {}
                Err(e) => {
                    self.message = format!("❌ Failed to play recording: {}", e);
                    self.show_message = true;
                }
            },
            _ => {}
        }

        Ok(DashboardAction::Continue)
    }

    async fn handle_dashboard_action(&mut self, action: DashboardAction) -> Result<()> {
        match action {
            DashboardAction::RecordAndTranscribe => {
                self.execute_record_and_transcribe().await?;
            }
            DashboardAction::AddExternalFile => {
                self.execute_add_external_file().await?;
            }
            DashboardAction::TranscribeSelected => {
                self.execute_transcribe_selected().await?;
            }
            _ => {}
        }
        Ok(())
    }

    fn load_recordings(&mut self) -> Result<()> {
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

        Ok(())
    }

    fn load_stats(&mut self) -> Result<()> {
        self.stats = Some(self.db.get_stats()?);
        Ok(())
    }

    async fn handle_search_input(&mut self, key_code: KeyCode) -> Result<DashboardAction> {
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

    fn next_recording(&mut self) {
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

    fn previous_recording(&mut self) {
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

    async fn next_page(&mut self) -> Result<()> {
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

    async fn previous_page(&mut self) -> Result<()> {
        if self.current_page > 0 {
            self.current_page -= 1;
            self.load_recordings()?;
        }
        Ok(())
    }

    async fn show_selected_transcript(&mut self) -> Result<()> {
        if let Some(selected) = self.table_state.selected() {
            if let Some(recording) = self.recordings.get(selected) {
                if recording.has_transcript {
                    match self.load_transcript_content(recording) {
                        Ok(content) => {
                            self.transcript_content = content;
                            self.show_transcript = true;
                        }
                        Err(e) => {
                            self.message = format!("❌ Failed to load transcript: {}", e);
                            self.show_message = true;
                        }
                    }
                } else {
                    self.message =
                        "❌ No transcript available for this recording. Use P to play instead."
                            .to_string();
                    self.show_message = true;
                }
            }
        }
        Ok(())
    }

    async fn play_selected_recording(&mut self) -> Result<()> {
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
                                self.message = format!("⚠️ mpv failed to play MP3: {}", e);
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
                        "▶ Playing: {}\nUsing player: {}\n\nPress ESC to stop playback",
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

    fn find_audio_file(&self, recording: &Recording) -> Option<PathBuf> {
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

    fn get_current_recording(&self) -> Option<Recording> {
        let selected_index = self.table_state.selected()?;
        self.recordings.get(selected_index).cloned()
    }

    fn show_delete_confirmation(&mut self) {
        if let Some(selected) = self.table_state.selected() {
            if let Some(recording) = self.recordings.get(selected).cloned() {
                self.delete_candidate = Some(recording);
                self.show_delete_confirm = true;
            }
        }
    }

    async fn handle_transcript_keys(&mut self, key_code: KeyCode) -> Result<DashboardAction> {
        match key_code {
            KeyCode::Up => {
                // Scroll up (decrease offset)
                if self.transcript_scroll_offset > 0 {
                    self.transcript_scroll_offset = self.transcript_scroll_offset.saturating_sub(1);
                }
                Ok(DashboardAction::Continue)
            }
            KeyCode::Down => {
                // Scroll down (increase offset)
                let content_width = 120; // Conservative estimate for terminal width
                let content_height = 25; // Conservative estimate for terminal height
                let wrapped_lines =
                    self.wrap_text_to_lines(&self.transcript_content, content_width);
                let max_scroll = wrapped_lines.len().saturating_sub(content_height);
                if self.transcript_scroll_offset < max_scroll {
                    self.transcript_scroll_offset += 1;
                }
                Ok(DashboardAction::Continue)
            }
            KeyCode::Char('c') | KeyCode::Char('C') => {
                // Copy transcript to clipboard
                match self.copy_transcript_to_clipboard() {
                    Ok(()) => {
                        self.message = "📋 Transcript copied to clipboard!".to_string();
                        self.show_message = true;
                    }
                    Err(e) => {
                        self.message = format!("❌ Failed to copy to clipboard: {}", e);
                        self.show_message = true;
                    }
                }
                Ok(DashboardAction::Continue)
            }
            KeyCode::PageUp => {
                // Page up (scroll up by larger amount)
                self.transcript_scroll_offset = self.transcript_scroll_offset.saturating_sub(10);
                Ok(DashboardAction::Continue)
            }
            KeyCode::PageDown => {
                // Page down (scroll down by larger amount)
                let content_width = 120; // Conservative estimate for terminal width
                let content_height = 25; // Conservative estimate for terminal height
                let wrapped_lines =
                    self.wrap_text_to_lines(&self.transcript_content, content_width);
                let max_offset = wrapped_lines.len().saturating_sub(content_height);
                self.transcript_scroll_offset =
                    std::cmp::min(self.transcript_scroll_offset + 10, max_offset);
                Ok(DashboardAction::Continue)
            }
            KeyCode::Home => {
                // Jump to top of transcript
                self.transcript_scroll_offset = 0;
                Ok(DashboardAction::Continue)
            }
            KeyCode::End => {
                // Jump to bottom of transcript
                let content_width = 120; // Conservative estimate for terminal width
                let content_height = 25; // Conservative estimate for terminal height
                let wrapped_lines =
                    self.wrap_text_to_lines(&self.transcript_content, content_width);
                if wrapped_lines.len() > content_height {
                    self.transcript_scroll_offset =
                        wrapped_lines.len().saturating_sub(content_height);
                } else {
                    self.transcript_scroll_offset = 0;
                }
                Ok(DashboardAction::Continue)
            }
            KeyCode::Char('g') => {
                // Jump to top of transcript (vim-style, alternative to Home)
                self.transcript_scroll_offset = 0;
                Ok(DashboardAction::Continue)
            }
            KeyCode::Char('G') => {
                // Jump to bottom of transcript (vim-style, alternative to End)
                let content_width = 120; // Conservative estimate for terminal width
                let content_height = 25; // Conservative estimate for terminal height
                let wrapped_lines =
                    self.wrap_text_to_lines(&self.transcript_content, content_width);
                if wrapped_lines.len() > content_height {
                    self.transcript_scroll_offset =
                        wrapped_lines.len().saturating_sub(content_height);
                } else {
                    self.transcript_scroll_offset = 0;
                }
                Ok(DashboardAction::Continue)
            }
            KeyCode::Char('b') => {
                // Page up (vim-style, alternative to PageUp)
                self.transcript_scroll_offset = self.transcript_scroll_offset.saturating_sub(10);
                Ok(DashboardAction::Continue)
            }
            KeyCode::Char('f') => {
                // Page down (vim-style, alternative to PageDown)
                let content_width = 120; // Conservative estimate for terminal width
                let content_height = 25; // Conservative estimate for terminal height
                let wrapped_lines =
                    self.wrap_text_to_lines(&self.transcript_content, content_width);
                let max_offset = wrapped_lines.len().saturating_sub(content_height);
                self.transcript_scroll_offset =
                    std::cmp::min(self.transcript_scroll_offset + 10, max_offset);
                Ok(DashboardAction::Continue)
            }
            KeyCode::Char('t') | KeyCode::Char('T') => {
                // Re-transcribe the current recording
                if let Some(recording) = self.get_current_recording() {
                    if self.transcription_task.is_some() {
                        self.message = "⚠️ Transcription already in progress".to_string();
                        self.show_message = true;
                        return Ok(DashboardAction::Continue);
                    }

                    // Close transcript view
                    self.show_transcript = false;
                    self.transcript_content.clear();
                    self.transcript_scroll_offset = 0;

                    // Start re-transcription using unified workflow
                    let transcription_mode = self.config.transcription.clone();
                    let directory_name = recording.directory_name.clone();
                    let display_name = recording
                        .display_name
                        .as_ref()
                        .unwrap_or(&recording.directory_name);

                    // Clean up display name for UI (remove _recording suffix)
                    let clean_display_name = if display_name == &recording.directory_name
                        && display_name.ends_with("_recording")
                    {
                        display_name
                            .strip_suffix("_recording")
                            .unwrap_or(display_name)
                    } else {
                        display_name
                    };

                    self.message = "🔄 Re-transcribing...".to_string();
                    self.show_message = true;

                    let model_info = match &self.config.transcription {
                        TranscriptionMode::Local { model_size } => {
                            format!("Local ({})", model_size)
                        }
                        TranscriptionMode::Api { .. } => "OpenAI API".to_string(),
                    };
                    self.progress_animation = Some(format!(
                        "🔄 Re-transcribing with {}: {}",
                        model_info, clean_display_name
                    ));
                    self.is_retranscribing = true;

                    // Use unified workflow for re-transcription
                    self.transcription_task = Some(tokio::spawn(async move {
                        let mut workflow = WorkflowManager::new().unwrap();
                        workflow
                            .retranscribe_recording_silent(&directory_name, transcription_mode)
                            .await
                    }));
                }
                Ok(DashboardAction::Continue)
            }
            KeyCode::Esc => {
                // Only Esc key closes the transcript (consistent behavior)
                self.show_transcript = false;
                self.transcript_content.clear();
                self.transcript_scroll_offset = 0;
                Ok(DashboardAction::Continue)
            }
            _ => {
                // Other keys are ignored (consistent behavior)
                Ok(DashboardAction::Continue)
            }
        }
    }

    async fn handle_settings_keys(&mut self, key_code: KeyCode) -> Result<DashboardAction> {
        match key_code {
            KeyCode::Esc => {
                self.current_view = DashboardView::Main;
                self.editing_api_key = false;
                Ok(DashboardAction::Continue)
            }
            KeyCode::Up => {
                if !self.editing_api_key {
                    match &self.config.transcription {
                        TranscriptionMode::Local { .. } => {
                            // Local mode: 0=Mode, 1=ModelSize
                            self.settings_selection = self.settings_selection.saturating_sub(1);
                        }
                        TranscriptionMode::Api { .. } => {
                            // API mode: 0=Mode, 1=APIKey
                            self.settings_selection = self.settings_selection.saturating_sub(1);
                        }
                    }
                }
                Ok(DashboardAction::Continue)
            }
            KeyCode::Down => {
                if !self.editing_api_key {
                    match &self.config.transcription {
                        TranscriptionMode::Local { .. } => {
                            // Local mode: 0=Mode, 1=ModelSize (max index 1)
                            self.settings_selection = std::cmp::min(self.settings_selection + 1, 1);
                        }
                        TranscriptionMode::Api { .. } => {
                            // API mode: 0=Mode, 1=APIKey (max index 1)
                            self.settings_selection = std::cmp::min(self.settings_selection + 1, 1);
                        }
                    }
                }
                Ok(DashboardAction::Continue)
            }
            KeyCode::Enter => {
                if self.editing_api_key {
                    // Save API key
                    let new_mode = TranscriptionMode::Api {
                        api_key: self.api_key_input.clone(),
                    };
                    match self.config.set_transcription_mode(new_mode) {
                        Ok(()) => {
                            self.config = ScribaConfig::load()?; // Reload config
                                                                 // Stay in settings, no success message
                        }
                        Err(e) => {
                            // Show error message only for actual errors
                            self.message = format!("❌ Failed to save API key: {}", e);
                            self.show_message = true;
                            self.return_to_view = Some(DashboardView::Settings);
                        }
                    }
                    self.editing_api_key = false;
                    self.api_key_input.clear();
                } else {
                    match self.settings_selection {
                        0 => {
                            // Toggle transcription mode
                            let new_mode = match &self.config.transcription {
                                TranscriptionMode::Local { .. } => {
                                    // Use preserved API key if available, otherwise empty
                                    let api_key = self
                                        .config
                                        .last_api_key
                                        .as_ref()
                                        .map(|key| key.clone())
                                        .unwrap_or_else(String::new);
                                    TranscriptionMode::Api { api_key }
                                }
                                TranscriptionMode::Api { .. } => TranscriptionMode::Local {
                                    model_size: LocalModelSize::Medium,
                                },
                            };
                            match self.config.set_transcription_mode(new_mode) {
                                Ok(()) => {
                                    self.config = ScribaConfig::load()?;
                                    // Reset selection to mode (index 0) when changing modes
                                    self.settings_selection = 0;
                                    // Stay in settings, no success message
                                }
                                Err(e) => {
                                    // Show error message only for actual errors
                                    self.message = format!("❌ Failed to change mode: {}", e);
                                    self.show_message = true;
                                    self.return_to_view = Some(DashboardView::Settings);
                                }
                            }
                        }
                        1 => {
                            match &self.config.transcription {
                                TranscriptionMode::Local { model_size } => {
                                    // In local mode: index 1 = Model Size
                                    let new_model = match model_size {
                                        LocalModelSize::Tiny => LocalModelSize::Base,
                                        LocalModelSize::Base => LocalModelSize::Small,
                                        LocalModelSize::Small => LocalModelSize::Medium,
                                        LocalModelSize::Medium => LocalModelSize::Large,
                                        LocalModelSize::Large => LocalModelSize::Turbo,
                                        LocalModelSize::Turbo => LocalModelSize::Tiny,
                                    };
                                    let new_mode = TranscriptionMode::Local {
                                        model_size: new_model,
                                    };
                                    match self.config.set_transcription_mode(new_mode) {
                                        Ok(()) => {
                                            self.config = ScribaConfig::load()?;
                                            // Stay in settings, no success message
                                        }
                                        Err(e) => {
                                            // Show error message only for actual errors
                                            self.message =
                                                format!("❌ Failed to change model: {}", e);
                                            self.show_message = true;
                                            self.return_to_view = Some(DashboardView::Settings);
                                        }
                                    }
                                }
                                TranscriptionMode::Api { .. } => {
                                    // In API mode: index 1 = API Key
                                    self.editing_api_key = true;
                                    self.api_key_input = match &self.config.transcription {
                                        TranscriptionMode::Api { api_key } => api_key.clone(),
                                        _ => String::new(),
                                    };
                                }
                            }
                        }
                        _ => {}
                    }
                }
                Ok(DashboardAction::Continue)
            }
            KeyCode::Char(c) => {
                if self.editing_api_key {
                    self.api_key_input.push(c);
                }
                Ok(DashboardAction::Continue)
            }
            KeyCode::Backspace => {
                if self.editing_api_key {
                    self.api_key_input.pop();
                }
                Ok(DashboardAction::Continue)
            }
            _ => Ok(DashboardAction::Continue),
        }
    }

    async fn handle_delete_confirmation(&mut self, key_code: KeyCode) -> Result<DashboardAction> {
        match key_code {
            KeyCode::Char('y') | KeyCode::Char('Y') => {
                // Confirm deletion
                if let Some(recording) = self.delete_candidate.take() {
                    match self.perform_delete_recording(recording).await {
                        Ok(()) => {}
                        Err(e) => {
                            self.message = format!("❌ Failed to delete recording: {}", e);
                            self.show_message = true;
                        }
                    }
                }
                self.show_delete_confirm = false;
                self.delete_candidate = None;
            }
            KeyCode::Char('n') | KeyCode::Char('N') | KeyCode::Esc => {
                // Cancel deletion
                self.show_delete_confirm = false;
                self.delete_candidate = None;
            }
            _ => {
                // Any other key, just ignore
            }
        }
        Ok(DashboardAction::Continue)
    }

    async fn handle_file_dialog_keys(&mut self, key_code: KeyCode) -> Result<DashboardAction> {
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
                            self.message = "❌ Please enter a file path".to_string();
                            self.show_message = true;
                            self.return_to_view = Some(DashboardView::Main);
                            self.show_file_dialog = false;
                            return Ok(DashboardAction::Continue);
                        }

                        // Check if file exists
                        let file_path = PathBuf::from(self.file_path_input.trim());
                        if !file_path.exists() {
                            self.message = "❌ File not found. Please check the path.".to_string();
                            self.show_message = true;
                            self.return_to_view = Some(DashboardView::Main);
                            self.show_file_dialog = false;
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

    async fn perform_delete_recording(&mut self, recording: Recording) -> Result<()> {
        if let Some(id) = recording.id {
            match self.db.delete_recording(id) {
                Ok(()) => {
                    let base_path = home_dir()
                        .context("Could not find home directory")?
                        .join("scriba_recordings");
                    let recording_dir = base_path.join(&recording.directory_name);

                    if recording_dir.exists() {
                        std::fs::remove_dir_all(&recording_dir).ok();
                    }

                    self.load_recordings()?;
                    self.load_stats()?;
                    self.message = "✅ Recording deleted successfully".to_string();
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

    fn load_transcript_content(&self, recording: &Recording) -> Result<String> {
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

    fn wrap_text_to_lines(&self, text: &str, max_width: usize) -> Vec<String> {
        let mut result = Vec::new();

        for line in text.lines() {
            if line.len() <= max_width {
                result.push(line.to_string());
            } else {
                // Split long lines into multiple wrapped lines
                let words: Vec<&str> = line.split_whitespace().collect();
                let mut current_line = String::new();

                for word in words {
                    if word.len() > max_width {
                        // Handle extremely long words by character breaking
                        if !current_line.is_empty() {
                            result.push(current_line);
                            current_line = String::new();
                        }

                        let chars: Vec<char> = word.chars().collect();
                        for chunk in chars.chunks(max_width) {
                            result.push(chunk.iter().collect());
                        }
                    } else {
                        let test_line = if current_line.is_empty() {
                            word.to_string()
                        } else {
                            format!("{} {}", current_line, word)
                        };

                        if test_line.len() <= max_width {
                            current_line = test_line;
                        } else {
                            result.push(current_line);
                            current_line = word.to_string();
                        }
                    }
                }

                if !current_line.is_empty() {
                    result.push(current_line);
                }
            }
        }

        // Handle edge case where text has no newlines at all
        if result.is_empty() && !text.is_empty() {
            let words: Vec<&str> = text.split_whitespace().collect();
            let mut current_line = String::new();

            for word in words {
                if word.len() > max_width {
                    if !current_line.is_empty() {
                        result.push(current_line);
                        current_line = String::new();
                    }

                    let chars: Vec<char> = word.chars().collect();
                    for chunk in chars.chunks(max_width) {
                        result.push(chunk.iter().collect());
                    }
                } else {
                    let test_line = if current_line.is_empty() {
                        word.to_string()
                    } else {
                        format!("{} {}", current_line, word)
                    };

                    if test_line.len() <= max_width {
                        current_line = test_line;
                    } else {
                        result.push(current_line);
                        current_line = word.to_string();
                    }
                }
            }

            if !current_line.is_empty() {
                result.push(current_line);
            }
        }

        result
    }

    fn copy_transcript_to_clipboard(&self) -> Result<()> {
        use arboard::Clipboard;

        let mut clipboard = Clipboard::new().context("Failed to access clipboard")?;

        clipboard
            .set_text(&self.transcript_content)
            .context("Failed to copy text to clipboard")?;

        Ok(())
    }

    fn ui(&mut self, f: &mut Frame) {
        match self.current_view {
            DashboardView::Main => self.render_main_dashboard(f),
            DashboardView::Help => self.render_help(f, f.size()),
            DashboardView::Settings => self.render_settings(f, f.size()),
        }
    }

    fn render_main_dashboard(&mut self, f: &mut Frame) {
        let size = f.size();

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

        // Main layout: Header + Content + Stats + Footer
        let main_chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(8), // Header (increased for ASCII art)
                Constraint::Min(6),    // Recordings Table
                Constraint::Length(4), // Statistics
                Constraint::Length(3), // Footer
            ])
            .split(size);

        // Header
        self.render_header(f, main_chunks[0]);

        // Table
        self.render_recordings_table(f, main_chunks[1]);

        // Statistics
        self.render_statistics(f, main_chunks[2]);

        // Footer
        self.render_footer(f, main_chunks[3]);

        // Search input overlay
        if self.search_mode {
            self.render_search_input(f, size);
        }
    }

    fn render_header(&self, f: &mut Frame, area: ratatui::layout::Rect) {
        let header_text = format!("{}\n🎵 RECORDING ON STEROIDS 🎵", ASCII_ART);

        let header = Paragraph::new(header_text)
            .style(
                Style::default()
                    .fg(Color::Magenta)
                    .add_modifier(Modifier::BOLD),
            )
            .alignment(Alignment::Center)
            .wrap(Wrap { trim: false })
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .style(Style::default().fg(Color::Cyan))
                    .border_type(ratatui::widgets::BorderType::Double),
            );
        f.render_widget(header, area);
    }

    fn render_recordings_table(&mut self, f: &mut Frame, area: ratatui::layout::Rect) {
        let header_cells = ["#", "Status", "Name", "Duration", "Model", "Created"]
            .iter()
            .map(|h| {
                Cell::from(*h).style(
                    Style::default()
                        .fg(Color::Yellow)
                        .add_modifier(Modifier::BOLD),
                )
            });

        let header = Row::new(header_cells)
            .style(Style::default())
            .height(1)
            .bottom_margin(1);

        let rows: Vec<Row> = self
            .recordings
            .iter()
            .enumerate()
            .map(|(i, recording)| {
                let display_name = recording
                    .display_name
                    .as_ref()
                    .unwrap_or(&recording.directory_name);

                let duration = recording
                    .duration_seconds
                    .map(|d| self.format_duration(d))
                    .unwrap_or_else(|| "Unknown".to_string());

                let status = if recording.has_transcript {
                    "[T]"
                } else {
                    "[A]"
                };
                let created = recording.created_at.format("%m/%d %H:%M").to_string();

                // Calculate global index across all pages
                let global_index = (self.current_page * self.page_size) + i + 1;

                // Format model used for display
                let model_display = if recording.has_transcript {
                    // Parse the model_used field to show a user-friendly format
                    match recording.model_used.as_str() {
                        // API models
                        "whisper-1" => "API",
                        s if s.starts_with("gpt") || s.contains("openai") => "API",
                        // Local models - extract size from format like "whisper-tiny", "whisper-turbo"
                        s if s.starts_with("whisper-") => {
                            let size = s.strip_prefix("whisper-").unwrap_or("");
                            match size {
                                "tiny" => "Tiny",
                                "base" => "Base",
                                "small" => "Small",
                                "medium" => "Medium",
                                "large" => "Large",
                                "large-v2" => "Large-v2",
                                "large-v3" => "Large-v3",
                                "turbo" => "Turbo",
                                _ => "Local",
                            }
                        }
                        // Legacy formats
                        "whisper" | "whisper.cpp" | "whisper-rs" => "Local",
                        // Unknown/empty
                        s if s.is_empty() => "-",
                        _ => &recording.model_used,
                    }
                } else {
                    "-"
                };

                let cells = vec![
                    Cell::from(global_index.to_string()),
                    Cell::from(status).style(if recording.has_transcript {
                        Style::default().fg(Color::Green)
                    } else {
                        Style::default().fg(Color::Blue)
                    }),
                    Cell::from(display_name.clone()),
                    Cell::from(duration),
                    Cell::from(model_display).style(if model_display == "API" {
                        Style::default().fg(Color::Cyan)
                    } else if model_display == "-" {
                        Style::default().fg(Color::Gray)
                    } else {
                        Style::default().fg(Color::Yellow)
                    }),
                    Cell::from(created),
                ];

                Row::new(cells).height(1).bottom_margin(0)
            })
            .collect();

        let table = Table::new(
            rows,
            [
                Constraint::Length(4),  // #
                Constraint::Length(8),  // Status
                Constraint::Min(15),    // Name (flexible)
                Constraint::Length(10), // Duration
                Constraint::Length(8),  // Model
                Constraint::Length(12), // Created
            ],
        )
        .header(header)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .style(Style::default().fg(Color::Cyan))
                .title({
                    let start_index = (self.current_page * self.page_size) + 1;
                    let end_index = start_index + self.recordings.len() - 1;
                    format!(
                        "Recordings (Page {} - #{}-#{})",
                        self.current_page + 1,
                        start_index,
                        end_index
                    )
                }),
        )
        .highlight_style(
            Style::default()
                .fg(Color::Black)
                .bg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        )
        .highlight_symbol("▶ ");

        f.render_stateful_widget(table, area, &mut self.table_state);
    }

    fn render_statistics(&self, f: &mut Frame, area: ratatui::layout::Rect) {
        let stats_text = if let Some(stats) = &self.stats {
            let transcribed_percentage = if stats.total_recordings > 0 {
                (stats.transcribed_count * 100) / stats.total_recordings
            } else {
                0
            };

            vec![
                Line::from(vec![
                    Span::styled("📊 Total: ", Style::default().fg(Color::White)),
                    Span::styled(
                        format!("{} recordings", stats.total_recordings),
                        Style::default()
                            .fg(Color::Yellow)
                            .add_modifier(Modifier::BOLD),
                    ),
                    Span::raw("    "),
                    Span::styled("🕒 Duration: ", Style::default().fg(Color::White)),
                    Span::styled(
                        stats.format_duration(),
                        Style::default()
                            .fg(Color::Green)
                            .add_modifier(Modifier::BOLD),
                    ),
                ]),
                Line::from(vec![
                    Span::styled("💾 Storage: ", Style::default().fg(Color::White)),
                    Span::styled(
                        stats.format_size(),
                        Style::default()
                            .fg(Color::Blue)
                            .add_modifier(Modifier::BOLD),
                    ),
                    Span::raw("      "),
                    Span::styled("📝 Transcribed: ", Style::default().fg(Color::White)),
                    Span::styled(
                        format!("{} ({}%)", stats.transcribed_count, transcribed_percentage),
                        Style::default()
                            .fg(Color::Magenta)
                            .add_modifier(Modifier::BOLD),
                    ),
                ]),
            ]
        } else {
            vec![Line::from("Loading statistics...")]
        };

        let stats = Paragraph::new(stats_text)
            .alignment(Alignment::Center)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title("STATISTICS")
                    .title_alignment(Alignment::Center)
                    .style(Style::default().fg(Color::Cyan)),
            );
        f.render_widget(stats, area);
    }

    fn render_footer(&self, f: &mut Frame, area: ratatui::layout::Rect) {
        let controls = if self.search_mode {
            "ESC: Cancel | ENTER: Search | Type to search..."
        } else {
            "↑↓: Navigate | [/]: Pages | ENTER: Transcript | P: Play | D/Del: Delete | /: Search | R/A/T: Quick Actions | S: Settings | H: Help | Q: Quit"
        };

        let controls_paragraph = Paragraph::new(controls)
            .style(Style::default().fg(Color::White))
            .alignment(Alignment::Center)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .style(Style::default().fg(Color::Blue))
                    .title("Controls"),
            );

        f.render_widget(controls_paragraph, area);
    }

    fn render_help(&self, f: &mut Frame, area: ratatui::layout::Rect) {
        let popup_area = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Percentage(15),
                Constraint::Percentage(70),
                Constraint::Percentage(15),
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

        let help_text = vec![
            Line::from("🎵 SCRIBA - RECORDING ON STEROIDS HELP 🎵"),
            Line::from(""),
            Line::from("Quick Actions:"),
            Line::from("  R          - Record Audio + Auto-Transcribe (Esc to stop)"),
            Line::from("  A          - Add External Audio File & Transcribe"),
            Line::from("  T          - Transcribe Selected Recording (also in transcript view)"),
            Line::from(""),
            Line::from("Navigation:"),
            Line::from("  ↑/↓        - Navigate recordings"),
            Line::from("  PgUp/PgDn  - Change pages (or '['/']')"),
            Line::from("  Enter      - View transcript"),
            Line::from("  P          - Play recording"),
            Line::from(""),
            Line::from("Actions:"),
            Line::from("  D          - Delete recording (with confirmation)"),
            Line::from("  /          - Search recordings"),
            Line::from("  S          - Settings (transcription mode, models)"),
            Line::from("  H/F1       - Show this help"),
            Line::from("  Q/Esc      - Quit"),
            Line::from(""),
            Line::from("Transcript Viewer:"),
            Line::from("  ↑/↓        - Scroll up/down"),
            Line::from("  PgUp/PgDn  - Page up/down (or 'b'/'f')"),
            Line::from("  Home/End   - Jump to top/bottom (or 'g'/'G')"),
            Line::from("  C          - Copy transcript to clipboard"),
            Line::from("  T          - Re-transcribe recording"),
            Line::from("  ESC        - Close transcript"),
            Line::from(""),
            Line::from("Recording Control:"),
            Line::from("  • Press R/A to start recording immediately"),
            Line::from("  • Press Esc during recording to stop and save"),
            Line::from("  • Real-time progress indicators for all operations"),
            Line::from(""),
            Line::from("Features:"),
            Line::from("  • Statistics always visible at bottom"),
            Line::from("  • Full-text search through transcripts"),
            Line::from("  • Integrated playback support"),
            Line::from("  • Direct recording from dashboard"),
            Line::from(""),
            Line::from("Press Esc to continue..."),
        ];

        let help_paragraph = Paragraph::new(help_text)
            .style(Style::default().fg(Color::White))
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .style(Style::default().fg(Color::Yellow))
                    .title("Help"),
            )
            .wrap(Wrap { trim: true });

        f.render_widget(help_paragraph, popup_area);
    }

    fn render_settings(&self, f: &mut Frame, area: ratatui::layout::Rect) {
        let popup_area = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Percentage(15),
                Constraint::Percentage(70),
                Constraint::Percentage(15),
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

        let mut settings_text = vec![
            Line::from(vec![Span::styled(
                "SETTINGS",
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            )]),
            Line::from(""),
        ];

        // Current transcription mode
        let mode_text = match &self.config.transcription {
            TranscriptionMode::Local { model_size } => {
                format!("⚙️ Local (Whisper {})  ← Press Enter to change", model_size)
            }
            TranscriptionMode::Api { .. } => "☁️ OpenAI API  ← Press Enter to change".to_string(),
        };

        let mode_style = if self.settings_selection == 0 {
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(Color::White)
        };

        settings_text.push(Line::from(vec![
            Span::styled("Transcription Mode: ", Style::default().fg(Color::Green)),
            Span::styled(mode_text, mode_style),
        ]));

        // Model size (only for local mode)
        if let TranscriptionMode::Local { model_size } = &self.config.transcription {
            let model_style = if self.settings_selection == 1 {
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(Color::White)
            };

            settings_text.push(Line::from(vec![
                Span::styled("Model Size: ", Style::default().fg(Color::Green)),
                Span::styled(
                    format!("{} ← Press Enter to cycle", model_size),
                    model_style,
                ),
            ]));
        }

        // Add mode-specific settings
        match &self.config.transcription {
            TranscriptionMode::Api { api_key } => {
                // API Mode: Show API Key configuration at index 1
                let api_key_display = if self.editing_api_key {
                    format!("{}_", self.api_key_input) // Show cursor
                } else {
                    if api_key.is_empty() {
                        "[Not Set] ← Press Enter to edit".to_string()
                    } else {
                        format!(
                            "{}****** ← Press Enter to edit",
                            &api_key[..api_key.len().min(4)]
                        )
                    }
                };

                let api_key_style = if self.settings_selection == 1 {
                    if self.editing_api_key {
                        Style::default()
                            .fg(Color::Green)
                            .add_modifier(Modifier::BOLD)
                    } else {
                        Style::default()
                            .fg(Color::Yellow)
                            .add_modifier(Modifier::BOLD)
                    }
                } else {
                    Style::default().fg(Color::White)
                };

                settings_text.push(Line::from(vec![
                    Span::styled("OpenAI API Key: ", Style::default().fg(Color::Green)),
                    Span::styled(api_key_display, api_key_style),
                ]));
            }
            TranscriptionMode::Local { .. } => {
                // Local Mode: No API Key shown, Model Size is already shown above
            }
        }

        // Audio settings preview
        settings_text.push(Line::from(""));
        settings_text.push(Line::from(vec![Span::styled(
            "Audio Settings:",
            Style::default().fg(Color::Green),
        )]));
        settings_text.push(Line::from(vec![Span::styled(
            format!(
                "  Sample Rate: {} Hz",
                self.config.audio_settings.sample_rate
            ),
            Style::default().fg(Color::Gray),
        )]));
        settings_text.push(Line::from(vec![Span::styled(
            format!("  Bitrate: {} kbps", self.config.audio_settings.bitrate),
            Style::default().fg(Color::Gray),
        )]));
        settings_text.push(Line::from(vec![Span::styled(
            format!("  Channels: {}", self.config.audio_settings.channels),
            Style::default().fg(Color::Gray),
        )]));
        settings_text.push(Line::from(vec![Span::styled(
            format!(
                "  Speech Optimized: {}",
                self.config.audio_settings.speech_optimized
            ),
            Style::default().fg(Color::Gray),
        )]));

        settings_text.push(Line::from(""));
        settings_text.push(Line::from(vec![Span::styled(
            "↑↓ Navigate  ⏎ Enter  ⎋ Esc",
            Style::default().fg(Color::Gray),
        )]));

        let settings_paragraph = Paragraph::new(settings_text)
            .style(Style::default().fg(Color::White))
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title("Settings")
                    .title_style(
                        Style::default()
                            .fg(Color::Cyan)
                            .add_modifier(Modifier::BOLD),
                    )
                    .border_style(Style::default().fg(Color::Cyan)),
            )
            .wrap(Wrap { trim: true });

        f.render_widget(settings_paragraph, popup_area);
    }

    fn render_message_popup(&self, f: &mut Frame, area: ratatui::layout::Rect) {
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

    fn render_transcript_popup(&self, f: &mut Frame, area: ratatui::layout::Rect) {
        let popup_area = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Percentage(10),
                Constraint::Percentage(80),
                Constraint::Percentage(10),
            ])
            .split(area)[1];

        let popup_area = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Percentage(5),
                Constraint::Percentage(90),
                Constraint::Percentage(5),
            ])
            .split(popup_area)[1];

        f.render_widget(Clear, popup_area);

        // Calculate available height and width for content (subtract borders)
        let content_height = popup_area.height.saturating_sub(2) as usize;
        let content_width = popup_area.width.saturating_sub(4) as usize; // Account for borders and padding

        // Handle text wrapping for very long lines
        let wrapped_lines = self.wrap_text_to_lines(&self.transcript_content, content_width);
        let total_lines = wrapped_lines.len();

        // Create scrollable content
        let (visible_content, scroll_info) = if total_lines > content_height {
            let max_scroll = total_lines.saturating_sub(content_height);
            let actual_offset = std::cmp::min(self.transcript_scroll_offset, max_scroll);
            let end = std::cmp::min(actual_offset + content_height, total_lines);

            let visible_lines = wrapped_lines[actual_offset..end].join("\n");
            let scroll_info = format!("📝 Transcript [Line {}/{}] - ↑↓/b/f/g/G: scroll, C: copy, T: re-transcribe, ESC: close",
                                    actual_offset + 1,
                                    total_lines);
            (visible_lines, scroll_info)
        } else {
            (
                self.transcript_content.clone(),
                "📝 Transcript - C: copy, T: re-transcribe, ESC: close".to_string(),
            )
        };

        let para = Paragraph::new(visible_content)
            .style(Style::default().fg(Color::White))
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .style(Style::default().fg(Color::Cyan))
                    .title(scroll_info),
            )
            .scroll((0, 0)); // Disable internal scrolling and wrapping since we handle it manually

        f.render_widget(para, popup_area);
    }

    fn render_delete_confirmation_popup(&self, f: &mut Frame, area: ratatui::layout::Rect) {
        let popup_area = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Percentage(25),
                Constraint::Length(12),
                Constraint::Percentage(63),
            ])
            .split(area)[1];

        let popup_area = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Percentage(25),
                Constraint::Percentage(50),
                Constraint::Percentage(25),
            ])
            .split(popup_area)[1];

        f.render_widget(Clear, popup_area);

        let recording_name = if let Some(recording) = &self.delete_candidate {
            recording
                .display_name
                .as_ref()
                .unwrap_or(&recording.directory_name)
                .clone()
        } else {
            "Unknown".to_string()
        };

        let confirmation_text = format!("⚠️  DELETE CONFIRMATION  ⚠️\n\nAre you sure you want to permanently delete:\n\n\"{}\"?\n\nThis action cannot be undone!\n\n[Y] Yes, delete it    [N] No, cancel\n[ESC] Cancel", recording_name);

        let para = Paragraph::new(confirmation_text)
            .style(Style::default().fg(Color::White))
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .style(Style::default().fg(Color::Red))
                    .title("⚠️  Confirm Deletion"),
            )
            .alignment(Alignment::Center)
            .wrap(Wrap { trim: true });

        f.render_widget(para, popup_area);
    }

    fn render_search_input(&self, f: &mut Frame, area: ratatui::layout::Rect) {
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

        let search_input = Paragraph::new(format!("🔍 Search: {}", self.search_query))
            .style(Style::default().fg(Color::Yellow))
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .style(Style::default().fg(Color::Cyan))
                    .title("Search Recordings"),
            );

        f.render_widget(search_input, popup_area);
    }

    async fn execute_record_and_transcribe(&mut self) -> Result<()> {
        // Check if already recording or transcribing
        if self.recording_task.is_some() || self.transcription_task.is_some() {
            self.message = "⚠️ Recording or transcription already in progress".to_string();
            self.show_message = true;
            return Ok(());
        }

        // Show immediate progress animation
        self.progress_animation = Some("🎙️ Recording... (Press Esc to stop)".to_string());
        self.progress_frame = 0;
        self.show_message = true;
        self.recording_mode = Some(RecordingMode::RecordAndTranscribe);

        // Generate filename and start recording task (no name prompt, consistent with A command)
        let recording_name = generate_recording_name(None);
        self.start_recording_task(recording_name).await?;

        Ok(())
    }

    async fn execute_add_external_file(&mut self) -> Result<()> {
        // Check if already recording or transcribing
        if self.recording_task.is_some() || self.transcription_task.is_some() {
            self.message = "⚠️ Recording or transcription already in progress".to_string();
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

    async fn start_file_import(&mut self, file_path: String, display_name: String) -> Result<()> {
        // Start animated progress and background task (non-blocking)
        self.progress_animation = Some(format!("📁 Importing + transcribing: {}", display_name));
        self.progress_frame = 0;
        self.show_message = true;
        self.is_importing = true;

        let source_path = PathBuf::from(file_path.trim());
        let transcription_mode = self.config.transcription.clone();
        let display_name_clone = display_name.clone();

        self.transcription_task = Some(tokio::spawn(async move {
            let mut workflow = WorkflowManager::new().unwrap();
            workflow
                .complete_import_workflow_silent(
                    &source_path,
                    Some(display_name_clone),
                    Some(transcription_mode),
                )
                .await
                .map(|_| ())
        }));

        Ok(())
    }

    // Removed unused execute_transcribe_file; dashboard uses TranscribeSelected (T) instead

    async fn execute_transcribe_selected(&mut self) -> Result<()> {
        // Check if transcription is already running
        if self.transcription_task.is_some() {
            self.message = "⚠️ Transcription already in progress. Please wait...".to_string();
            self.show_message = true;
            return Ok(());
        }

        // Get the selected recording
        let selected_index = match self.table_state.selected() {
            Some(i) => i,
            None => {
                self.message = "❌ No recording selected".to_string();
                self.show_message = true;
                return Ok(());
            }
        };

        let selected_recording = match self.recordings.get(selected_index) {
            Some(recording) => recording.clone(),
            None => {
                self.message = "❌ Invalid recording selection".to_string();
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
                    "⚠️ Recording already has transcript. Press T again to overwrite.".to_string();
                self.show_message = true;
                return Ok(());
            }
        } else {
            // Clear any previous warning state
            self.last_transcribe_warning = None;
        }

        // No API key required in local transcription mode
        let display_name = selected_recording
            .display_name
            .as_ref()
            .unwrap_or(&selected_recording.directory_name);

        // Clean up display name for UI (remove _recording suffix)
        let clean_display_name = if display_name == &selected_recording.directory_name
            && display_name.ends_with("_recording")
        {
            display_name
                .strip_suffix("_recording")
                .unwrap_or(display_name)
        } else {
            display_name
        };

        // Show immediate progress animation
        let model_info = match &self.config.transcription {
            TranscriptionMode::Local { model_size } => {
                format!("Local ({})", model_size)
            }
            TranscriptionMode::Api { .. } => "OpenAI API".to_string(),
        };
        let action = if has_transcript {
            "Re-transcribing"
        } else {
            "Transcribing"
        };
        self.progress_animation = Some(format!(
            "🔄 {} with {}: {}",
            action, model_info, clean_display_name
        ));
        self.progress_frame = 0;
        self.show_message = true;

        // Start transcription using unified workflow
        let directory_name = selected_recording.directory_name.clone();
        let transcription_mode = self.config.transcription.clone();

        self.transcription_task = Some(tokio::spawn(async move {
            let mut workflow = WorkflowManager::new().unwrap();
            workflow
                .retranscribe_recording_silent(&directory_name, transcription_mode)
                .await
        }));

        Ok(())
    }

    fn format_duration(&self, seconds: i64) -> String {
        let hours = seconds / 3600;
        let minutes = (seconds % 3600) / 60;
        let secs = seconds % 60;

        if hours > 0 {
            format!("{}h {}m", hours, minutes)
        } else if minutes > 0 {
            format!("{}m {}s", minutes, secs)
        } else {
            format!("{}s", secs)
        }
    }

    async fn create_stereo_temp_file(
        &self,
        mono_file_path: &std::path::Path,
    ) -> Result<std::path::PathBuf> {
        use std::fs;

        // Create a temporary file path for the stereo version
        let temp_dir = std::env::temp_dir();
        let temp_filename = format!(
            "scriba_stereo_{}.wav",
            mono_file_path
                .file_stem()
                .unwrap_or_default()
                .to_string_lossy()
        );
        let temp_path = temp_dir.join(temp_filename);

        // Use Rust's hound crate to convert mono to stereo
        let mono_reader =
            hound::WavReader::open(mono_file_path).context("Failed to open mono audio file")?;

        let spec = mono_reader.spec();

        // Create stereo spec (2 channels)
        let stereo_spec = hound::WavSpec {
            channels: 2,
            sample_rate: spec.sample_rate,
            bits_per_sample: spec.bits_per_sample,
            sample_format: spec.sample_format,
        };

        let mut stereo_writer = hound::WavWriter::create(&temp_path, stereo_spec)
            .context("Failed to create stereo audio file")?;

        // Convert samples based on format
        match spec.sample_format {
            hound::SampleFormat::Float => {
                // 32-bit float samples
                for sample in mono_reader.into_samples::<f32>() {
                    match sample {
                        Ok(s) => {
                            // Write the same sample to both left and right channels
                            stereo_writer.write_sample(s)?; // Left
                            stereo_writer.write_sample(s)?; // Right
                        }
                        Err(e) => {
                            return Err(anyhow::anyhow!("Error processing audio sample: {}", e))
                        }
                    }
                }
            }
            hound::SampleFormat::Int => {
                // Integer samples (16-bit or 24-bit)
                if spec.bits_per_sample == 16 {
                    for sample in mono_reader.into_samples::<i16>() {
                        match sample {
                            Ok(s) => {
                                stereo_writer.write_sample(s)?; // Left
                                stereo_writer.write_sample(s)?; // Right
                            }
                            Err(e) => {
                                return Err(anyhow::anyhow!(
                                    "Error processing audio sample: {}",
                                    e
                                ));
                            }
                        }
                    }
                } else if spec.bits_per_sample == 24 {
                    for sample in mono_reader.into_samples::<i32>() {
                        match sample {
                            Ok(s) => {
                                stereo_writer.write_sample(s)?; // Left
                                stereo_writer.write_sample(s)?; // Right
                            }
                            Err(e) => {
                                return Err(anyhow::anyhow!(
                                    "Error processing audio sample: {}",
                                    e
                                ));
                            }
                        }
                    }
                } else {
                    return Err(anyhow::anyhow!(
                        "Unsupported bit depth: {}",
                        spec.bits_per_sample
                    ));
                }
            }
        }

        // Finalize the stereo file
        stereo_writer
            .finalize()
            .context("Failed to finalize stereo audio file")?;

        // Schedule cleanup of temp file after a delay
        let temp_path_clone = temp_path.clone();
        tokio::spawn(async move {
            tokio::time::sleep(tokio::time::Duration::from_secs(30)).await;
            let _ = fs::remove_file(&temp_path_clone);
        });

        Ok(temp_path)
    }

    fn emergency_stop_all_audio_players(&self) -> Result<()> {
        // Kill all common audio players as a fallback when PID is not available
        #[cfg(unix)]
        {
            use std::process::Command;
            // Try to kill common audio players
            let players = ["mpv", "ffplay", "afplay"];
            for player in &players {
                let _ = Command::new("killall").arg(player).output();
            }
        }

        #[cfg(windows)]
        {
            use std::process::Command;
            // Try to kill common audio players on Windows
            let players = ["mpv.exe", "ffplay.exe"];
            for player in &players {
                let _ = Command::new("taskkill")
                    .arg("/IM")
                    .arg(player)
                    .arg("/F")
                    .output();
            }
        }
        Ok(())
    }

    fn stop_audio_playback(&self, pid: u32) -> Result<()> {
        #[cfg(unix)]
        {
            use std::process::Command;
            // Use SIGTERM first for graceful shutdown, then SIGKILL if needed
            let _ = Command::new("kill")
                .arg("-TERM")
                .arg(pid.to_string())
                .output();

            // Give a very brief moment for graceful shutdown
            std::thread::sleep(std::time::Duration::from_millis(10));

            // Use SIGKILL immediately for faster termination (audio players can be stubborn)
            let kill_result = Command::new("kill")
                .arg("-KILL")
                .arg(pid.to_string())
                .output();

            // Also try pkill in case the process spawned children
            let _ = Command::new("pkill")
                .arg("-P")
                .arg(pid.to_string())
                .output();

            match kill_result {
                Ok(_) => Ok(()),
                Err(_e) => {
                    // If direct kill fails, try killall on common audio players
                    let _ = Command::new("killall")
                        .arg("mpv")
                        .arg("ffplay")
                        .arg("afplay")
                        .output();
                    Ok(())
                }
            }
        }

        #[cfg(windows)]
        {
            use std::process::Command;
            let _ = Command::new("taskkill")
                .arg("/PID")
                .arg(pid.to_string())
                .arg("/F")
                .output();
            Ok(())
        }
    }

    // start_progress_animation not used; progress is updated directly via fields

    fn stop_progress_animation(&mut self) {
        self.progress_animation = None;
    }

    fn update_progress_message(&mut self) {
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

    fn create_volume_bar(&self, level: f32) -> String {
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

    async fn start_recording_task(&mut self, recording_name: String) -> Result<()> {
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

        // Use unified recording function with TUI control channels
        // record_audio and RecordOptions are already imported from crate::core
        let output_path = PathBuf::from(&recording_name);

        self.recording_task = Some(tokio::spawn(async move {
            record_audio(
                output_path,
                RecordOptions {
                    compression_settings: Some(compression_settings),
                    stop_rx: Some(stop_rx),
                    level_tx: Some(level_tx),
                    verbose: false,
                },
            )
            .await
        }));

        Ok(())
    }

    fn render_file_dialog_popup(&self, f: &mut Frame, area: ratatui::layout::Rect) {
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
                            .fg(Color::Cyan)
                            .add_modifier(Modifier::BOLD),
                    )
                    .border_style(Style::default().fg(Color::Cyan)),
            )
            .wrap(Wrap { trim: true });

        f.render_widget(dialog_paragraph, popup_area);
    }
}
