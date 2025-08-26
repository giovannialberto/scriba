use crate::database::{Database, Recording, RecordingStats};
use crate::record::record;
use crate::transcribe::transcribe_file;
use crate::audio::{CompressionSettings, AudioFormat};
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
    widgets::{
        Block, Borders, Clear, Paragraph, Wrap, Row, Table, TableState, Cell,
    },
    Frame, Terminal,
};
use std::env;
use std::io::{self, Write};
use std::path::PathBuf;
use std::time::Duration;
use chrono::Local;
use tokio::process::Command as TokioCommand;
use dirs::home_dir;
use anyhow::Context;

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
}

#[derive(Debug, PartialEq)]
enum DashboardView {
    Main,
    Help,
}

#[derive(Debug)]
enum DashboardAction {
    Continue,
    Quit,
    RecordAndTranscribe,
    RecordOnly,
    TranscribeFile,
}

impl Dashboard {
    pub fn new() -> Result<Self> {
        let db = Database::new()?;
        let mut table_state = TableState::default();
        table_state.select(Some(0));

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

    async fn run_app<B: ratatui::backend::Backend>(&mut self, terminal: &mut Terminal<B>) -> Result<()> {
        loop {
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
        // If audio is playing, any key press stops it
        if let Some(pid) = self.current_playback_pid {
            self.stop_audio_playback(pid)?;
            self.current_playback_pid = None;
            self.message = "🛑 Audio playback stopped".to_string();
            self.show_message = true;
            return Ok(DashboardAction::Continue);
        }
        
        if self.show_message {
            // Any key press closes the message popup
            self.show_message = false;
            self.message.clear();
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
            KeyCode::Char('r') | KeyCode::Char('R') => {
                return Ok(DashboardAction::RecordAndTranscribe);
            }
            KeyCode::Char('a') | KeyCode::Char('A') => {
                return Ok(DashboardAction::RecordOnly);
            }
            KeyCode::Char('t') | KeyCode::Char('T') => {
                return Ok(DashboardAction::TranscribeFile);
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
            KeyCode::PageUp => {
                self.previous_page().await?;
            }
            KeyCode::PageDown => {
                self.next_page().await?;
            }
            KeyCode::Enter => {
                match self.show_selected_transcript().await {
                    Ok(()) => {}
                    Err(e) => {
                        self.message = format!("❌ Failed to load transcript: {}", e);
                        self.show_message = true;
                    }
                }
            }
            KeyCode::Char('d') => {
                self.show_delete_confirmation();
            }
            KeyCode::Delete => {
                self.show_delete_confirmation();
            }
            KeyCode::Char('p') | KeyCode::Char('P') => {
                match self.play_selected_recording().await {
                    Ok(()) => {}
                    Err(e) => {
                        self.message = format!("❌ Failed to play recording: {}", e);
                        self.show_message = true;
                    }
                }
            }
            _ => {}
        }

        Ok(DashboardAction::Continue)
    }

    async fn handle_dashboard_action(&mut self, action: DashboardAction) -> Result<()> {
        match action {
            DashboardAction::RecordAndTranscribe => {
                self.execute_record_and_transcribe().await?;
            }
            DashboardAction::RecordOnly => {
                self.execute_record_only().await?;
            }
            DashboardAction::TranscribeFile => {
                self.execute_transcribe_file().await?;
            }
            _ => {}
        }
        Ok(())
    }

    fn load_recordings(&mut self) -> Result<()> {
        let offset = (self.current_page * self.page_size) as i64;
        
        self.recordings = if self.search_query.is_empty() {
            self.db.list_recordings(Some(self.page_size as i64), Some(offset))?
        } else {
            let search_results = self.db.search_transcripts(&self.search_query, None)?;
            search_results.into_iter().map(|(recording, _)| recording).collect()
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
            },
            KeyCode::Enter => {
                self.search_mode = false;
                self.load_recordings()?;
            },
            KeyCode::Backspace => {
                self.search_query.pop();
            },
            KeyCode::Char(c) => {
                self.search_query.push(c);
            },
            _ => {},
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
        if self.recordings.len() == self.page_size {
            self.current_page += 1;
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
                    self.message = "❌ No transcript available for this recording. Use P to play instead.".to_string();
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
                let is_mp3 = audio_path.extension()
                    .and_then(|ext| ext.to_str())
                    .map(|ext| ext.to_lowercase() == "mp3")
                    .unwrap_or(false);

                // Candidate players differ by platform. For MP3 files, prioritize mpv/ffplay over afplay
                #[cfg(target_os = "macos")]
                let candidates: Vec<(&str, &[&str])> = if is_mp3 {
                    vec![
                        ("mpv", &["--really-quiet", "--audio-channels=stereo"]),
                        ("ffplay", &["-nodisp", "-autoexit", "-loglevel", "quiet", "-ac", "2"]),
                        ("afplay", &[]),         // Last resort for MP3
                    ]
                } else {
                    vec![
                        ("mpv", &["--really-quiet", "--audio-channels=stereo"]),
                        ("ffplay", &["-nodisp", "-autoexit", "-loglevel", "quiet", "-ac", "2"]),
                        ("afplay", &[]),         // Works well with WAV
                    ]
                };

                #[cfg(all(unix, not(target_os = "macos")))]
                let candidates: Vec<(&str, &[&str])> = vec![
                    ("mpv", &["--really-quiet", "--audio-channels=stereo"]),
                    ("ffplay", &["-nodisp", "-autoexit", "-loglevel", "quiet", "-ac", "2"]),
                    ("aplay", &["-c", "2"]),    // Force stereo output
                ];

                #[cfg(target_os = "windows")]
                let candidates: Vec<(&str, &[&str])> = vec![
                    ("mpv", &["--really-quiet", "--audio-channels=stereo"]),
                    ("ffplay", &["-nodisp", "-autoexit", "-loglevel", "quiet", "-ac", "2"]),
                    ("powershell", &["-NoProfile", "-Command", "(New-Object Media.SoundPlayer '" ]), // will be handled specially
                ];

                // Try each candidate until one spawns successfully
                let mut launched_with: Option<String> = None;

                #[cfg(not(target_os = "windows"))]
                for (prog, base_args) in candidates {
                    let mut cmd = TokioCommand::new(prog);
                    
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
                        for a in base_args { cmd.arg(a); }
                        cmd.arg(&audio_path);
                    }
                    
                    match cmd.spawn() {
                        Ok(mut child) => { 
                            launched_with = Some(prog.to_string()); 
                            
                            // Store child process for potential termination
                            let child_id = child.id();
                            tokio::spawn(async move {
                                let _ = child.wait().await;
                            });
                            
                            // Store the process ID for killing on key press
                            self.current_playback_pid = child_id;
                            break; 
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
                    for (prog, base_args) in &candidates[..candidates.len()-1] { // All except powershell
                        let mut cmd = TokioCommand::new(prog);
                        for a in base_args { cmd.arg(a); }
                        cmd.arg(&audio_path);
                        match cmd.spawn() {
                            Ok(_child) => { launched_with = Some(prog.to_string()); break; }
                            Err(_e) => continue,
                        }
                    }
                    
                    // PowerShell SoundPlayer fallback if no other player worked
                    if launched_with.is_none() {
                        let escaped = audio_path.to_string_lossy().replace("'", "''");
                        let ps = format!(
                            "$p=New-Object Media.SoundPlayer '{}';$p.Play();",
                            escaped
                        );
                        match TokioCommand::new("powershell")
                            .arg("-NoProfile")
                            .arg("-Command")
                            .arg(ps)
                            .spawn()
                        {
                            Ok(_child) => { launched_with = Some("powershell".to_string()); }
                            Err(_e) => {}
                        }
                    }
                }

                if let Some(player) = launched_with {
                    let name = recording
                        .display_name
                        .as_ref()
                        .unwrap_or(&recording.directory_name);
                    self.message = format!("▶ Playing: {}\nUsing player: {}\n\n(Playback runs externally; return here when done.)", name, player);
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
            } else { Ok(()) }
        } else { Ok(()) }
    }

    fn find_audio_file(&self, recording: &Recording) -> Option<PathBuf> {
        let base_path = home_dir()?.join("scriba_recordings").join(&recording.directory_name);
        if !base_path.exists() { return None; }
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
            },
            KeyCode::Down => {
                // Scroll down (increase offset) 
                let content_width = 120; // Conservative estimate for terminal width
                let content_height = 25; // Conservative estimate for terminal height
                let wrapped_lines = self.wrap_text_to_lines(&self.transcript_content, content_width);
                let max_scroll = wrapped_lines.len().saturating_sub(content_height);
                if self.transcript_scroll_offset < max_scroll {
                    self.transcript_scroll_offset += 1;
                }
                Ok(DashboardAction::Continue)
            },
            KeyCode::Char('c') | KeyCode::Char('C') => {
                // Copy transcript to clipboard
                match self.copy_transcript_to_clipboard() {
                    Ok(()) => {
                        self.message = "📋 Transcript copied to clipboard!".to_string();
                        self.show_message = true;
                    },
                    Err(e) => {
                        self.message = format!("❌ Failed to copy to clipboard: {}", e);
                        self.show_message = true;
                    }
                }
                Ok(DashboardAction::Continue)
            },
            KeyCode::PageUp => {
                // Page up (scroll up by larger amount)
                self.transcript_scroll_offset = self.transcript_scroll_offset.saturating_sub(10);
                Ok(DashboardAction::Continue)
            },
            KeyCode::PageDown => {
                // Page down (scroll down by larger amount)
                let content_width = 120; // Conservative estimate for terminal width
                let content_height = 25; // Conservative estimate for terminal height
                let wrapped_lines = self.wrap_text_to_lines(&self.transcript_content, content_width);
                let max_offset = wrapped_lines.len().saturating_sub(content_height);
                self.transcript_scroll_offset = std::cmp::min(
                    self.transcript_scroll_offset + 10,
                    max_offset
                );
                Ok(DashboardAction::Continue)
            },
            KeyCode::Home => {
                // Jump to top of transcript
                self.transcript_scroll_offset = 0;
                Ok(DashboardAction::Continue)
            },
            KeyCode::End => {
                // Jump to bottom of transcript
                let content_width = 120; // Conservative estimate for terminal width
                let content_height = 25; // Conservative estimate for terminal height
                let wrapped_lines = self.wrap_text_to_lines(&self.transcript_content, content_width);
                if wrapped_lines.len() > content_height {
                    self.transcript_scroll_offset = wrapped_lines.len().saturating_sub(content_height);
                } else {
                    self.transcript_scroll_offset = 0;
                }
                Ok(DashboardAction::Continue)
            },
            KeyCode::Char('g') => {
                // Jump to top of transcript (vim-style, alternative to Home)
                self.transcript_scroll_offset = 0;
                Ok(DashboardAction::Continue)
            },
            KeyCode::Char('G') => {
                // Jump to bottom of transcript (vim-style, alternative to End)
                let content_width = 120; // Conservative estimate for terminal width
                let content_height = 25; // Conservative estimate for terminal height
                let wrapped_lines = self.wrap_text_to_lines(&self.transcript_content, content_width);
                if wrapped_lines.len() > content_height {
                    self.transcript_scroll_offset = wrapped_lines.len().saturating_sub(content_height);
                } else {
                    self.transcript_scroll_offset = 0;
                }
                Ok(DashboardAction::Continue)
            },
            KeyCode::Char('b') => {
                // Page up (vim-style, alternative to PageUp)
                self.transcript_scroll_offset = self.transcript_scroll_offset.saturating_sub(10);
                Ok(DashboardAction::Continue)
            },
            KeyCode::Char('f') => {
                // Page down (vim-style, alternative to PageDown)
                let content_width = 120; // Conservative estimate for terminal width
                let content_height = 25; // Conservative estimate for terminal height
                let wrapped_lines = self.wrap_text_to_lines(&self.transcript_content, content_width);
                let max_offset = wrapped_lines.len().saturating_sub(content_height);
                self.transcript_scroll_offset = std::cmp::min(
                    self.transcript_scroll_offset + 10,
                    max_offset
                );
                Ok(DashboardAction::Continue)
            },
            _ => {
                // Any other key closes the transcript
                self.show_transcript = false;
                self.transcript_content.clear();
                self.transcript_scroll_offset = 0;
                Ok(DashboardAction::Continue)
            }
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
            },
            KeyCode::Char('n') | KeyCode::Char('N') | KeyCode::Esc => {
                // Cancel deletion
                self.show_delete_confirm = false;
                self.delete_candidate = None;
            },
            _ => {
                // Any other key, just ignore
            }
        }
        Ok(DashboardAction::Continue)
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
            return Err(anyhow::anyhow!("Selected recording has no database ID; cannot delete."));
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
        
        // Fallback: try to load from file
        let base_path = home_dir()
            .context("Could not find home directory")?
            .join("scriba_recordings");
        let transcript_path = base_path.join(&recording.directory_name).join("transcript.txt");
        
        if transcript_path.exists() {
            std::fs::read_to_string(&transcript_path)
                .context("Failed to read transcript file")
        } else {
            Err(anyhow::anyhow!("Transcript file not found"))
        }
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
        
        let mut clipboard = Clipboard::new()
            .context("Failed to access clipboard")?;
        
        clipboard.set_text(&self.transcript_content)
            .context("Failed to copy text to clipboard")?;
        
        Ok(())
    }

    fn ui(&mut self, f: &mut Frame) {
        match self.current_view {
            DashboardView::Main => self.render_main_dashboard(f),
            DashboardView::Help => self.render_help(f, f.size()),
        }
    }

    fn render_main_dashboard(&mut self, f: &mut Frame) {
        let size = f.size();

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
                Constraint::Length(3),  // Header
                Constraint::Min(6),     // Recordings Table
                Constraint::Length(4),  // Statistics
                Constraint::Length(3),  // Footer
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
        let header = Paragraph::new("🎵 SCRIBA - RECORDING LIBRARY WITH STATISTICS 🎵")
            .style(Style::default().fg(Color::Magenta).add_modifier(Modifier::BOLD))
            .alignment(Alignment::Center)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .style(Style::default().fg(Color::Cyan))
                    .border_type(ratatui::widgets::BorderType::Double),
            );
        f.render_widget(header, area);
    }

    fn render_recordings_table(&mut self, f: &mut Frame, area: ratatui::layout::Rect) {
        let header_cells = ["#", "Status", "Name", "Duration", "Created"]
            .iter()
            .map(|h| Cell::from(*h).style(Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)));
        
        let header = Row::new(header_cells)
            .style(Style::default())
            .height(1)
            .bottom_margin(1);

        let rows: Vec<Row> = self.recordings.iter().enumerate().map(|(i, recording)| {
            let display_name = recording.display_name.as_ref()
                .unwrap_or(&recording.directory_name);
            
            let duration = recording.duration_seconds
                .map(|d| self.format_duration(d))
                .unwrap_or_else(|| "Unknown".to_string());

            let status = if recording.has_transcript { "[T]" } else { "[A]" };
            let created = recording.created_at.format("%m/%d %H:%M").to_string();

            let cells = vec![
                Cell::from((i + 1).to_string()),
                Cell::from(status).style(
                    if recording.has_transcript {
                        Style::default().fg(Color::Green)
                    } else {
                        Style::default().fg(Color::Blue)
                    }
                ),
                Cell::from(display_name.clone()),
                Cell::from(duration),
                Cell::from(created),
            ];

            Row::new(cells).height(1).bottom_margin(0)
        }).collect();

        let table = Table::new(
            rows,
            [
                Constraint::Length(4),   // #
                Constraint::Length(8),   // Status
                Constraint::Min(20),     // Name (flexible)
                Constraint::Length(12),  // Duration
                Constraint::Length(12),  // Created
            ]
        )
            .header(header)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .style(Style::default().fg(Color::Cyan))
                    .title("Recordings")
            )
            .highlight_style(
                Style::default()
                    .fg(Color::Black)
                    .bg(Color::Cyan)
                    .add_modifier(Modifier::BOLD)
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
                    Span::styled(format!("{} recordings", stats.total_recordings), Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
                    Span::raw("    "),
                    Span::styled("🕒 Duration: ", Style::default().fg(Color::White)),
                    Span::styled(stats.format_duration(), Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)),
                ]),
                Line::from(vec![
                    Span::styled("💾 Storage: ", Style::default().fg(Color::White)),
                    Span::styled(stats.format_size(), Style::default().fg(Color::Blue).add_modifier(Modifier::BOLD)),
                    Span::raw("      "),
                    Span::styled("📝 Transcribed: ", Style::default().fg(Color::White)),
                    Span::styled(format!("{} ({}%)", stats.transcribed_count, transcribed_percentage), Style::default().fg(Color::Magenta).add_modifier(Modifier::BOLD)),
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
            "↑↓: Navigate | ENTER: Transcript | P: Play | D/Del: Delete | /: Search | R/A/T: Quick Actions | H: Help | Q: Quit"
        };

        let controls_paragraph = Paragraph::new(controls)
            .style(Style::default().fg(Color::White))
            .alignment(Alignment::Center)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .style(Style::default().fg(Color::Blue))
                    .title("Controls")
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
            Line::from("🎵 SCRIBA - RECORDING LIBRARY WITH STATISTICS HELP 🎵"),
            Line::from(""),
            Line::from("Quick Actions:"),
            Line::from("  R          - Record Audio + Auto-Transcribe"),
            Line::from("  A          - Record Audio Only"),
            Line::from("  T          - Transcribe Existing File"),
            Line::from(""),
            Line::from("Navigation:"),
            Line::from("  ↑/↓        - Navigate recordings"),
            Line::from("  PgUp/PgDn  - Change pages"),
            Line::from("  Enter      - View transcript"),
            Line::from("  P          - Play recording"),
            Line::from(""),
            Line::from("Actions:"),
            Line::from("  D          - Delete recording (with confirmation)"),
            Line::from("  /          - Search recordings"),
            Line::from("  H/F1       - Show this help"),
            Line::from("  Q/Esc      - Quit"),
            Line::from(""),
            Line::from("Transcript Viewer:"),
            Line::from("  ↑/↓        - Scroll up/down"),
            Line::from("  PgUp/PgDn  - Page up/down (or 'b'/'f')"),
            Line::from("  Home/End   - Jump to top/bottom (or 'g'/'G')"),
            Line::from("  C          - Copy transcript to clipboard"),
            Line::from("  ESC        - Close transcript"),
            Line::from(""),
            Line::from("Features:"),
            Line::from("  • Statistics always visible at bottom"),
            Line::from("  • Full-text search through transcripts"),
            Line::from("  • Integrated playback support"),
            Line::from("  • Quick recording actions available"),
            Line::from(""),
            Line::from("Press any key to continue..."),
        ];

        let help_paragraph = Paragraph::new(help_text)
            .style(Style::default().fg(Color::White))
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .style(Style::default().fg(Color::Yellow))
                    .title("Help")
            )
            .wrap(Wrap { trim: true });

        f.render_widget(help_paragraph, popup_area);
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
                    .title("Message (press any key)")
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
            let scroll_info = format!("📝 Transcript [Line {}/{}] - ↑↓/b/f/g/G: scroll, C: copy, ESC: close", 
                                    actual_offset + 1, 
                                    total_lines);
            (visible_lines, scroll_info)
        } else {
            (self.transcript_content.clone(), "📝 Transcript - C: copy, ESC: close".to_string())
        };
        
        let para = Paragraph::new(visible_content)
            .style(Style::default().fg(Color::White))
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .style(Style::default().fg(Color::Cyan))
                    .title(scroll_info)
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
            recording.display_name.as_ref()
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
                    .title("⚠️  Confirm Deletion")
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
                    .title("Search Recordings")
            );

        f.render_widget(search_input, popup_area);
    }

    async fn execute_record_and_transcribe(&mut self) -> Result<()> {
        // Temporarily restore terminal for input
        disable_raw_mode()?;
        
        print!("\n📝 Enter recording name (optional, press Enter to skip): ");
        io::stdout().flush()?;
        let mut name_input = String::new();
        io::stdin().read_line(&mut name_input)?;
        let name_input = name_input.trim();
        
        let name = if name_input.is_empty() { None } else { Some(name_input.to_string()) };
        
        // Get API key
        let api_key = match env::var("OPENAI_API_KEY") {
            Ok(key) => key,
            Err(_) => {
                print!("🔑 Enter OpenAI API key: ");
                io::stdout().flush()?;
                let mut api_input = String::new();
                io::stdin().read_line(&mut api_input)?;
                api_input.trim().to_string()
            }
        };
        
        if api_key.is_empty() {
            println!("❌ API key required for transcription. Operation cancelled.");
            print!("Press Enter to continue...");
            io::stdout().flush()?;
            let mut _input = String::new();
            io::stdin().read_line(&mut _input)?;
            enable_raw_mode()?;
            return Ok(());
        }
        
        println!("\n🎙️ Starting recording session...");
        
        // Generate filename
        let recording_name = self.generate_filename(name);
        let audio_output = PathBuf::from(&recording_name);
        
        // Record audio
        // Use speech-optimized WAV compression (device adaptation happens in record function)
        let compression_settings = CompressionSettings::speech_optimized();
        let record_result = record(audio_output.clone(), Some(compression_settings)).await;
        
        if record_result.is_ok() {
            println!("\n📝 Recording complete! Starting transcription...");
            
            match transcribe_file(&audio_output, &audio_output, &api_key).await {
                Ok(()) => {
                    println!("✅ Transcription complete!");
                    println!("📁 Files saved in: ~/scriba_recordings/{}/", recording_name);
                }
                Err(err) => {
                    println!("❌ Transcription failed: {}", err);
                }
            }
        } else if let Err(err) = record_result {
            println!("❌ Recording failed: {}", err);
        }
        
        print!("\nPress Enter to continue...");
        io::stdout().flush()?;
        let mut _input = String::new();
        io::stdin().read_line(&mut _input)?;
        
        // Restore terminal for TUI
        enable_raw_mode()?;
        
        // Reload dashboard data
        self.load_recordings()?;
        self.load_stats()?;
        
        Ok(())
    }
    
    async fn execute_record_only(&mut self) -> Result<()> {
        // Temporarily restore terminal for input
        disable_raw_mode()?;
        
        print!("\n📝 Enter recording name (optional, press Enter to skip): ");
        io::stdout().flush()?;
        let mut name_input = String::new();
        io::stdin().read_line(&mut name_input)?;
        let name_input = name_input.trim();
        
        let name = if name_input.is_empty() { None } else { Some(name_input.to_string()) };
        
        println!("\n🔴 Starting recording session...");
        
        // Generate filename
        let recording_name = self.generate_filename(name);
        let audio_output = PathBuf::from(&recording_name);
        
        // Record audio
        // Use speech-optimized WAV compression (device adaptation happens in record function)
        let compression_settings = CompressionSettings::speech_optimized();
        match record(audio_output, Some(compression_settings)).await {
            Ok(()) => {
                println!("✅ Recording complete!");
                println!("📁 File saved in: ~/scriba_recordings/{}/", recording_name);
            }
            Err(err) => {
                println!("❌ Recording failed: {}", err);
            }
        }
        
        print!("\nPress Enter to continue...");
        io::stdout().flush()?;
        let mut _input = String::new();
        io::stdin().read_line(&mut _input)?;
        
        // Restore terminal for TUI
        enable_raw_mode()?;
        
        // Reload dashboard data
        self.load_recordings()?;
        self.load_stats()?;
        
        Ok(())
    }
    
    async fn execute_transcribe_file(&mut self) -> Result<()> {
        // Temporarily restore terminal for input
        disable_raw_mode()?;
        
        print!("\n📁 Enter path to audio file: ");
        io::stdout().flush()?;
        let mut path_input = String::new();
        io::stdin().read_line(&mut path_input)?;
        let input_path = path_input.trim();
        
        if input_path.is_empty() {
            println!("❌ File path required. Operation cancelled.");
            print!("Press Enter to continue...");
            io::stdout().flush()?;
            let mut _input = String::new();
            io::stdin().read_line(&mut _input)?;
            enable_raw_mode()?;
            return Ok(());
        }
        
        print!("📝 Enter transcript name (optional, press Enter to skip): ");
        io::stdout().flush()?;
        let mut name_input = String::new();
        io::stdin().read_line(&mut name_input)?;
        let name_input = name_input.trim();
        
        let name = if name_input.is_empty() { None } else { Some(name_input.to_string()) };
        
        // Get API key
        let api_key = match env::var("OPENAI_API_KEY") {
            Ok(key) => key,
            Err(_) => {
                print!("🔑 Enter OpenAI API key: ");
                io::stdout().flush()?;
                let mut api_input = String::new();
                io::stdin().read_line(&mut api_input)?;
                api_input.trim().to_string()
            }
        };
        
        if api_key.is_empty() {
            println!("❌ API key required for transcription. Operation cancelled.");
            print!("Press Enter to continue...");
            io::stdout().flush()?;
            let mut _input = String::new();
            io::stdin().read_line(&mut _input)?;
            enable_raw_mode()?;
            return Ok(());
        }
        
        // Generate output filename
        let output = if let Some(n) = name {
            let timestamp = Local::now().format("%Y-%m-%d_%H-%M-%S");
            let sanitized = n.replace(' ', "-").replace(['/', '\\', ':', '*', '?', '"', '<', '>', '|'], "_");
            PathBuf::from(format!("{}_{}_transcript", timestamp, sanitized))
        } else {
            let timestamp = Local::now().format("%Y-%m-%d_%H-%M-%S");
            PathBuf::from(format!("{}_transcript", timestamp))
        };
        
        println!("\n📝 Starting transcription...");
        
        match transcribe_file(&PathBuf::from(input_path), &output, &api_key).await {
            Ok(()) => {
                println!("✅ Transcription complete!");
                println!("📁 File saved in: ~/scriba_recordings/{}/", output.display());
            }
            Err(err) => {
                println!("❌ Transcription failed: {}", err);
            }
        }
        
        print!("\nPress Enter to continue...");
        io::stdout().flush()?;
        let mut _input = String::new();
        io::stdin().read_line(&mut _input)?;
        
        // Restore terminal for TUI
        enable_raw_mode()?;
        
        // Reload dashboard data
        self.load_recordings()?;
        self.load_stats()?;
        
        Ok(())
    }
    
    fn generate_filename(&self, name: Option<String>) -> String {
        let timestamp = Local::now().format("%Y-%m-%d_%H-%M-%S");
        match name {
            Some(n) => {
                let sanitized = n.replace(' ', "-").replace(['/', '\\', ':', '*', '?', '"', '<', '>', '|'], "_");
                format!("{}_{}", timestamp, sanitized)
            },
            None => format!("{}_recording", timestamp),
        }
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
    
    async fn create_stereo_temp_file(&self, mono_file_path: &std::path::Path) -> Result<std::path::PathBuf> {
        use std::fs;
        
        // Create a temporary file path for the stereo version
        let temp_dir = std::env::temp_dir();
        let temp_filename = format!("scriba_stereo_{}.wav", 
            mono_file_path.file_stem().unwrap_or_default().to_string_lossy());
        let temp_path = temp_dir.join(temp_filename);
        
        // Use Rust's hound crate to convert mono to stereo
        let mono_reader = hound::WavReader::open(mono_file_path)
            .context("Failed to open mono audio file")?;
        
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
                            stereo_writer.write_sample(s)?;  // Left
                            stereo_writer.write_sample(s)?;  // Right
                        }
                        Err(e) => return Err(anyhow::anyhow!("Error processing audio sample: {}", e)),
                    }
                }
            }
            hound::SampleFormat::Int => {
                // Integer samples (16-bit or 24-bit)
                if spec.bits_per_sample == 16 {
                    for sample in mono_reader.into_samples::<i16>() {
                        match sample {
                            Ok(s) => {
                                stereo_writer.write_sample(s)?;  // Left
                                stereo_writer.write_sample(s)?;  // Right
                            }
                            Err(e) => {
                                return Err(anyhow::anyhow!("Error processing audio sample: {}", e));
                            }
                        }
                    }
                } else if spec.bits_per_sample == 24 {
                    for sample in mono_reader.into_samples::<i32>() {
                        match sample {
                            Ok(s) => {
                                stereo_writer.write_sample(s)?;  // Left
                                stereo_writer.write_sample(s)?;  // Right
                            }
                            Err(e) => {
                                return Err(anyhow::anyhow!("Error processing audio sample: {}", e));
                            }
                        }
                    }
                } else {
                    return Err(anyhow::anyhow!("Unsupported bit depth: {}", spec.bits_per_sample));
                }
            }
        }
        
        // Finalize the stereo file
        stereo_writer.finalize()
            .context("Failed to finalize stereo audio file")?;
        
        // Schedule cleanup of temp file after a delay
        let temp_path_clone = temp_path.clone();
        tokio::spawn(async move {
            tokio::time::sleep(tokio::time::Duration::from_secs(30)).await;
            let _ = fs::remove_file(&temp_path_clone);
        });
        
        Ok(temp_path)
    }
    
    
    fn stop_audio_playback(&self, pid: u32) -> Result<()> {
        #[cfg(unix)]
        {
            use std::process::Command;
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
                Err(e) => {
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
}