use crate::database::{Database, Recording};
use anyhow::{Context, Result};
use std::io::{self, stdout, Write};
use dirs::home_dir;
use crossterm::{
    style::{Color, Stylize}, 
    terminal::{Clear, ClearType, enable_raw_mode, disable_raw_mode},
    cursor::{MoveTo, Hide, Show},
    event::{self, Event, KeyCode, KeyEvent},
    execute,
};
use std::thread;
use std::time::Duration;

pub struct RecordingLibrary {
    db: Database,
    current_page: usize,
    page_size: usize,
    recordings: Vec<Recording>,
    selected_index: usize,
}

impl RecordingLibrary {
    pub fn new() -> Result<Self> {
        let db = Database::new()?;
        Ok(Self {
            db,
            current_page: 0,
            page_size: 10,
            recordings: Vec::new(),
            selected_index: 0,
        })
    }

    pub async fn show_library(&mut self) -> Result<()> {
        // Enable raw mode for arrow key detection
        enable_raw_mode()?;
        // Hide cursor for better visual experience
        execute!(stdout(), Hide)?;
        
        // Initial load
        self.show_loading_animation()?;
        self.load_recordings()?;

        if self.recordings.is_empty() {
            self.show_empty_library();
            disable_raw_mode()?;
            execute!(stdout(), Show)?;
            return Ok(());
        }
        
        loop {
            self.display_library()?;
            
            match self.get_user_choice()? {
                LibraryChoice::Quit => break,
                LibraryChoice::NextPage => {
                    self.next_page();
                    self.load_recordings()?;
                }
                LibraryChoice::PrevPage => {
                    self.prev_page();
                    self.load_recordings()?;
                }
                LibraryChoice::Play(index) => {
                    if let Some(recording) = self.recordings.get(index) {
                        self.play_recording(recording)?;
                    }
                }
                LibraryChoice::Delete(index) => {
                    if let Some(recording) = self.recordings.get(index) {
                        self.delete_recording(recording.clone())?;
                        self.load_recordings()?;
                    }
                }
                LibraryChoice::ShowStats => {
                    self.show_stats()?;
                }
                LibraryChoice::Search => {
                    self.search_recordings().await?;
                }
                LibraryChoice::MoveUp | LibraryChoice::MoveDown => {
                    // Navigation handled in get_user_choice, just refresh display
                }
                // Current selection handled via Enter mapping to Play(self.selected_index)
            }
        }

        // Restore terminal state when exiting
        disable_raw_mode()?;
        execute!(stdout(), Show)?;
        Ok(())
    }


    fn show_loading_animation(&self) -> Result<()> {
        execute!(stdout(), Clear(ClearType::All), MoveTo(0, 0))?;
        
        let frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
        let colors = [Color::Cyan, Color::Magenta, Color::Yellow, Color::Green];
        
        for i in 0..12 {
            let frame = frames[i % frames.len()];
            let color = colors[i % colors.len()];
            
            execute!(stdout(), MoveTo(25, 10))?;
            print!("{} {} {}", 
                   frame.with(color), 
                   "LOADING RECORDINGS".with(Color::White).bold(),
                   frame.with(color));
            stdout().flush()?;
            thread::sleep(Duration::from_millis(80));
        }
        
        Ok(())
    }

    fn load_recordings(&mut self) -> Result<()> {
        let offset = (self.current_page * self.page_size) as i64;
        self.recordings = self.db.list_recordings(Some(self.page_size as i64), Some(offset))?;
        
        // Reset selected index when loading new recordings
        self.selected_index = 0;
        if self.selected_index >= self.recordings.len() && !self.recordings.is_empty() {
            self.selected_index = self.recordings.len() - 1;
        }
        
        Ok(())
    }

    fn display_library(&self) -> Result<()> {
        execute!(stdout(), Clear(ClearType::All), MoveTo(0, 0))?;

        // Retro header with colorful gradient effect
        println!("{}", "╔════════════════════════════════════════════════════════╗".with(Color::Cyan).bold());
        println!("{}", "║                   🎵 RECORDING LIBRARY 🎵                ║".with(Color::Magenta).bold());
        println!("{}", "╚════════════════════════════════════════════════════════╝".with(Color::Cyan).bold());
        println!();

        // Show current page info with retro styling
        let total_recordings = self.recordings.len();
        let start_index = self.current_page * self.page_size + 1;
        let end_index = std::cmp::min(start_index + total_recordings - 1, start_index + self.page_size - 1);
        
        if total_recordings > 0 {
            let page_info = format!("【 PAGE {} 】 Showing {}-{} recordings", 
                    self.current_page + 1, start_index, end_index);
            println!("{}", page_info.with(Color::Yellow).bold());
            println!();
        }

        // Table header for proper alignment
        println!("{}", format!("    {}  {:<25} │ {:<8} │ {}", 
                 "ST", "NAME", "DURATION", "CREATED").with(Color::Yellow).bold());
        println!("{}", "─────────────────────────────────────────────────────────".with(Color::DarkGrey));

        // Display recordings with fixed-width formatting for perfect alignment
        for (i, recording) in self.recordings.iter().enumerate() {
            let is_selected = i == self.selected_index;
            let number = i + 1;
            let display_name = recording.display_name.as_ref()
                .unwrap_or(&recording.directory_name);
            
            let duration = recording.duration_seconds
                .map(|d| self.format_duration(d))
                .unwrap_or_else(|| "Unknown".to_string());

            // Use text-based status indicators for consistent alignment
            let status_icon = if recording.has_transcript { "[T]" } else { "[A]" };
            
            let created_date = recording.created_at.format("%m/%d %H:%M").to_string();
            
            // Fixed-width formatting ensures perfect column alignment
            if is_selected {
                let selected_line = format!("▶{:2}. {} {:<25} │ {:<8} │ {}", 
                    number, status_icon, display_name, duration, created_date);
                println!("{}", selected_line.with(Color::Black).on(Color::Cyan).bold());
            } else {
                let normal_line = format!(" {:2}. {} {:<25} │ {:<8} │ {}", 
                    number, status_icon, display_name, duration, created_date);
                println!("{}", normal_line.with(Color::White));
            }
        }

        println!();

        // Retro-styled command bar with pixelated look
        println!("{}", "▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓".with(Color::DarkBlue));
        
        let controls = vec![
            ("↑/↓", "Navigate", Color::Green),
            ("ENTER", "Play", Color::Yellow),
            ("D", "Delete", Color::Red),
            ("S", "Stats", Color::Magenta),
            ("/", "Search", Color::Cyan),
            ("N/P", "Page", Color::Blue),
            ("Q", "Quit", Color::White),
        ];
        
        print!("┃ ");
        for (i, (key, desc, color)) in controls.iter().enumerate() {
            print!("{}{} {}", 
                   key.with(*color).bold(),
                   ":".with(Color::DarkGrey),
                   desc.with(Color::White));
            if i < controls.len() - 1 {
                print!(" {} ", "│".with(Color::DarkGrey));
            }
        }
        println!(" ┃");
        
        println!("{}", "▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓".with(Color::DarkBlue));
        
        print!("{} ", ">>".with(Color::Green).bold());
        stdout().flush()?;
        
        Ok(())
    }




    fn show_empty_library(&self) {
        // Temporarily disable raw mode for user input
        disable_raw_mode().unwrap();
        execute!(stdout(), Show).unwrap();
        
        println!("╔════════════════════════════════════════════════════════╗");
        println!("║                   🎵 RECORDING LIBRARY 🎵                ║");
        println!("╚════════════════════════════════════════════════════════╝");
        println!();
        println!("{}", "📭 EMPTY LIBRARY".with(Color::Yellow).bold());
        println!();
        println!("{}", "You haven't made any recordings yet!".with(Color::White));
        println!("{}", "Go back to the main menu to start your first recording session.".with(Color::Cyan));
        println!();
        println!("Press Enter to continue...");
        
        let mut input = String::new();
        io::stdin().read_line(&mut input).unwrap();
    }

    fn get_user_choice(&mut self) -> Result<LibraryChoice> {
        loop {
            if event::poll(std::time::Duration::from_millis(100))? {
                if let Event::Key(KeyEvent { code, .. }) = event::read()? {
                    match code {
                        KeyCode::Up => {
                            if self.selected_index > 0 {
                                self.selected_index -= 1;
                                return Ok(LibraryChoice::MoveUp);
                            }
                        }
                        KeyCode::Down => {
                            if self.selected_index < self.recordings.len().saturating_sub(1) {
                                self.selected_index += 1;
                                return Ok(LibraryChoice::MoveDown);
                            }
                        }
                        KeyCode::Enter => {
                            return Ok(LibraryChoice::Play(self.selected_index));
                        }
                        KeyCode::Char('q') | KeyCode::Char('Q') | KeyCode::Esc => {
                            return Ok(LibraryChoice::Quit);
                        }
                        KeyCode::Char('s') | KeyCode::Char('S') => {
                            return Ok(LibraryChoice::ShowStats);
                        }
                        KeyCode::Char('/') => {
                            return Ok(LibraryChoice::Search);
                        }
                        KeyCode::Char('n') | KeyCode::Char('N') => {
                            return Ok(LibraryChoice::NextPage);
                        }
                        KeyCode::Char('p') | KeyCode::Char('P') => {
                            return Ok(LibraryChoice::PrevPage);
                        }
                        KeyCode::Char('d') | KeyCode::Char('D') => {
                            return Ok(LibraryChoice::Delete(self.selected_index));
                        }
                        KeyCode::Char(c) => {
                            // Try to parse as number for direct selection
                            if let Some(num) = c.to_digit(10) {
                                let index = (num as usize).saturating_sub(1);
                                if index < self.recordings.len() {
                                    self.selected_index = index;
                                    return Ok(LibraryChoice::Play(index));
                                }
                            }
                        }
                        _ => {}
                    }
                }
            }
        }
    }


    fn next_page(&mut self) {
        if self.recordings.len() == self.page_size {
            self.current_page += 1;
            self.selected_index = 0; // Reset selection to top of new page
        }
    }

    fn prev_page(&mut self) {
        if self.current_page > 0 {
            self.current_page -= 1;
            self.selected_index = 0; // Reset selection to top of new page
        }
    }

    fn play_recording(&self, recording: &Recording) -> Result<()> {
        // Temporarily disable raw mode for user input
        disable_raw_mode()?;
        execute!(stdout(), Show)?;
        
        println!("\n🎵 Playing: {}", 
                 recording.display_name.as_ref().unwrap_or(&recording.directory_name));
        
        let base_path = home_dir()
            .context("Could not find home directory")?
            .join("scriba_recordings");
        let audio_path = base_path.join(&recording.directory_name).join("recording.wav");
        
        if !audio_path.exists() {
            println!("❌ Audio file not found: {}", audio_path.display());
            println!("Press Enter to continue...");
            let mut input = String::new();
            io::stdin().read_line(&mut input)?;
            return Ok(());
        }

        // For now, just show file info. In the future, we'll add actual playback
        println!("📁 Location: {}", audio_path.display());
        if let Some(duration) = recording.duration_seconds {
            println!("⏱️  Duration: {}", self.format_duration(duration));
        }
        if recording.has_transcript {
            println!("📝 Transcript available");
            
            // Show transcript if it exists
            let transcript_path = base_path.join(&recording.directory_name).join("transcript.txt");
            if transcript_path.exists() {
                if let Ok(content) = std::fs::read_to_string(&transcript_path) {
                    println!();
                    println!("──── TRANSCRIPT ────");
                    let lines: Vec<&str> = content.lines().collect();
                    for line in lines.iter().take(5) {
                        println!("{}", line);
                    }
                    if lines.len() > 5 {
                        println!("... ({} more lines)", lines.len() - 5);
                    }
                    println!("────────────────────");
                }
            }
        }

        println!("\n💡 Future: Audio playback will be implemented here!");
        println!("Press Enter to continue...");
        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        
        // Re-enable raw mode
        enable_raw_mode()?;
        execute!(stdout(), Hide)?;
        
        Ok(())
    }

    fn delete_recording(&mut self, recording: Recording) -> Result<()> {
        // Temporarily disable raw mode for user input
        disable_raw_mode()?;
        execute!(stdout(), Show)?;
        
        println!("\n⚠️  Are you sure you want to delete this recording?");
        println!("   📁 {}", recording.display_name.as_ref().unwrap_or(&recording.directory_name));
        print!("   Type 'yes' to confirm: ");
        stdout().flush()?;
        
        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        
        if input.trim().to_lowercase() == "yes" {
            // Delete from database
            if let Some(id) = recording.id {
                self.db.delete_recording(id)?;
                
                // Delete files
                let base_path = home_dir()
                    .context("Could not find home directory")?
                    .join("scriba_recordings");
                let recording_dir = base_path.join(&recording.directory_name);
                
                if recording_dir.exists() {
                    std::fs::remove_dir_all(&recording_dir)?;
                }
                
                println!("✅ Recording deleted successfully!");
            }
        } else {
            println!("❌ Deletion cancelled.");
        }
        
        println!("Press Enter to continue...");
        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        
        // Re-enable raw mode
        enable_raw_mode()?;
        execute!(stdout(), Hide)?;
        
        Ok(())
    }

    fn show_stats(&self) -> Result<()> {
        // Temporarily disable raw mode for user input
        disable_raw_mode()?;
        execute!(stdout(), Show)?;
        
        let stats = self.db.get_stats()?;
        
        println!();
        println!("╔════════════════════════════════════════════════════════╗");
        println!("║                RECORDING STATISTICS                    ║");
        println!("╚════════════════════════════════════════════════════════╝");
        println!();
        println!("📊 Total Recordings: {}", stats.total_recordings);
        println!("⏱️  Total Duration: {}", stats.format_duration());
        println!("💾 Total Storage: {}", stats.format_size());
        println!("📝 Transcribed: {}", stats.transcribed_count);
        println!("🔤 Total Words: {}", stats.total_words);
        if stats.total_recordings > 0 {
            let avg_duration = stats.total_duration_seconds / stats.total_recordings;
            println!("📊 Avg Duration: {}", self.format_duration(avg_duration));
        }
        
        println!("\n💡 Your personal knowledge base is growing!");
        println!("Press Enter to continue...");
        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        
        // Re-enable raw mode
        enable_raw_mode()?;
        execute!(stdout(), Hide)?;
        
        Ok(())
    }

    async fn search_recordings(&mut self) -> Result<()> {
        // Temporarily disable raw mode for user input
        disable_raw_mode()?;
        execute!(stdout(), Show)?;
        
        print!("\n🔍 Search recordings (enter keywords): ");
        stdout().flush()?;
        
        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let query = input.trim();
        
        if query.is_empty() {
            return Ok(());
        }

        println!("🔍 Searching for '{}'...", query);
        
        match self.db.search_transcripts(query, Some(10)) {
            Ok(results) => {
                if results.is_empty() {
                    println!("❌ No recordings found matching '{}'", query);
                } else {
                    println!("✅ Found {} matching recording(s):", results.len());
                    println!();
                    
                    for (i, (recording, _transcript)) in results.iter().enumerate() {
                        println!("{}. 🎙️  {}", 
                                 i + 1, 
                                 recording.display_name.as_ref().unwrap_or(&recording.directory_name));
                        println!("   📅 {}", recording.created_at.format("%Y-%m-%d %H:%M"));
                        println!();
                    }
                }
            }
            Err(e) => {
                println!("❌ Search failed: {}", e);
                println!("💡 Make sure you have transcribed recordings to search through.");
            }
        }
        
        println!("Press Enter to continue...");
        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        
        // Re-enable raw mode
        enable_raw_mode()?;
        execute!(stdout(), Hide)?;
        
        Ok(())
    }

    fn format_duration(&self, seconds: i64) -> String {
        let hours = seconds / 3600;
        let minutes = (seconds % 3600) / 60;
        let secs = seconds % 60;
        
        if hours > 0 {
            format!("{}h {}m {}s", hours, minutes, secs)
        } else if minutes > 0 {
            format!("{}m {}s", minutes, secs)
        } else {
            format!("{}s", secs)
        }
    }
}

#[derive(Debug)]
enum LibraryChoice {
    Quit,
    NextPage,
    PrevPage,
    Play(usize),
    Delete(usize),
    ShowStats,
    Search,
    MoveUp,
    MoveDown,
}
