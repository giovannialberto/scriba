use std::path::PathBuf;
use anyhow::Context;
use reqwest::Client;
use reqwest::multipart::{Form, Part};
use std::fs::File;
use std::io::{BufWriter, Write};
use lazy_static::lazy_static;
use dirs::home_dir;
use std::time::{Duration, Instant};
use tokio::time::sleep;
use std::io::stdout;
use serde_json::Value;
use unicode_width::UnicodeWidthStr;
use crate::database::{Database, Transcript};
use chrono::Utc;

lazy_static! {
    static ref BASE_PATH: PathBuf = home_dir().expect("error home dir").join("scriba_recordings");
}

struct TranscriptionProgress {
    start_time: Instant,
    animation_frame: usize,
}

impl TranscriptionProgress {
    fn new() -> Self {
        Self {
            start_time: Instant::now(),
            animation_frame: 0,
        }
    }

    async fn show_progress(&mut self) {
        let elapsed = self.start_time.elapsed().as_secs();
        
        // Rotating spinner animation
        let spinner_chars = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
        let spinner = spinner_chars[self.animation_frame % spinner_chars.len()];
        
        // Progress messages that change over time
        let message = match elapsed {
            0..=3 => "Uploading audio file",
            4..=8 => "OpenAI is processing your audio",
            9..=15 => "Converting speech to text",
            16..=25 => "This is taking longer than usual",
            _ => "Almost there, hang tight"
        };
        
        // Show elapsed time
        let time_display = if elapsed < 60 {
            format!("{}s", elapsed)
        } else {
            format!("{}m {}s", elapsed / 60, elapsed % 60)
        };
        
        // Animated progress bar
        let bar_width = 30;
        let progress_pos = (elapsed as usize * 2) % (bar_width * 2);
        let mut bar = vec![' '; bar_width];
        
        if progress_pos < bar_width {
            for i in 0..=progress_pos.min(bar_width - 1) {
                bar[i] = if i == progress_pos { '█' } else { '▓' };
            }
        } else {
            let reverse_pos = (bar_width * 2 - 1) - progress_pos;
            for i in reverse_pos..bar_width {
                bar[i] = if i == reverse_pos { '█' } else { '▓' };
            }
        }
        
        let bar_str: String = bar.into_iter().collect();
        
        print!("\r🎵 {} [{}] {} - {}", spinner, bar_str, message, time_display);
        stdout().flush().unwrap();
        
        self.animation_frame += 1;
        sleep(Duration::from_millis(100)).await;
    }
    
}

async fn show_transcription_typing_effect(text: &str) {
    const BOX_WIDTH: usize = 58;
    const CONTENT_WIDTH: usize = BOX_WIDTH - 4; // Account for "│ " and " │"
    
    println!("\n╭─ TRANSCRIPTION RESULT ─────────────────────────────────╮");
    
    // Split text into words and wrap properly
    let words: Vec<&str> = text.split_whitespace().collect();
    let mut current_line = String::new();
    let mut lines = Vec::new();
    
    for word in words {
        if current_line.is_empty() {
            current_line = word.to_string();
        } else {
            let test_line = format!("{} {}", current_line, word);
            if test_line.width() <= CONTENT_WIDTH {
                current_line = test_line;
            } else {
                lines.push(current_line);
                current_line = word.to_string();
            }
        }
    }
    if !current_line.is_empty() {
        lines.push(current_line);
    }
    
    // Show completion message without typewriter delay
    let success_msg = "✅ Transcription completed successfully!";
    let success_width = success_msg.width();
    let success_padding = CONTENT_WIDTH.saturating_sub(success_width);
    println!("│ {}{} │", success_msg, " ".repeat(success_padding));
    
    println!("│{}  │", " ".repeat(CONTENT_WIDTH));
    
    let dashboard_msg = "📋 Use dashboard to view and copy transcripts";
    let dashboard_width = dashboard_msg.width();
    let dashboard_padding = CONTENT_WIDTH.saturating_sub(dashboard_width);
    println!("│ {}{} │", dashboard_msg, " ".repeat(dashboard_padding));
    
    println!("╰────────────────────────────────────────────────────────╯");
}

// Enhanced transcription function with delightful UX
pub async fn transcribe_file(
    input_path: &PathBuf,
    output_path: &PathBuf,
    api_key: &str,
) -> Result<(), anyhow::Error> {
    // Find the audio file (could be WAV or MP3)
    let recording_dir = BASE_PATH.join(input_path);
    let (audio_file_path, filename) = if recording_dir.join("recording.mp3").exists() {
        let path = recording_dir.join("recording.mp3");
        (path, "recording.mp3".to_string())
    } else if recording_dir.join("recording.wav").exists() {
        let path = recording_dir.join("recording.wav");
        (path, "recording.wav".to_string())
    } else {
        return Err(anyhow::anyhow!("No audio file found (recording.wav or recording.mp3) in {}", recording_dir.display()));
    };
    
    let audio_file = std::fs::read(&audio_file_path)
        .context("Unable to read the input file")?;

    // Create progress indicator
    let mut progress = TranscriptionProgress::new();
    
    // Show initial progress
    println!("\n🎤 → 📝 Transcribing your audio...\n");

    // Set up the reqwest client
    let client = Client::new();

    // Create a multipart form with the audio file and model parameter
    let form = Form::new()
        .part("file", Part::bytes(audio_file).file_name(filename))
        .text("model", "whisper-1");

    // Start the progress animation in background
    let progress_task = tokio::spawn(async move {
        loop {
            progress.show_progress().await;
        }
    });

    // Make the POST request to the OpenAI API
    let response = client
        .post("https://api.openai.com/v1/audio/transcriptions")
        .multipart(form)
        .header("Authorization", format!("Bearer {}", api_key))
        .send()
        .await
        .context("Failed to send request")?;
    
    // Stop the progress animation
    progress_task.abort();
    
    // Clear progress line
    print!("\r{}", " ".repeat(80));
    print!("\r");
    stdout().flush().unwrap();

    if response.status().is_success() {
        let text = response.text().await.context("Failed to read response body")?;

        // Parse the JSON response to extract just the text
        let json: Value = serde_json::from_str(&text)?;
        let transcription_text = json["text"].as_str().unwrap_or(&text);

        // Save the transcript to the specified or default file
        let transcript_file_path = BASE_PATH.join(output_path).join("transcript.txt");
        
        // Ensure the directory exists
        if let Some(parent) = transcript_file_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        
        let transcript_file = File::create(&transcript_file_path)?;
        let mut transcript_writer = BufWriter::new(transcript_file);
        transcript_writer.write_all(transcription_text.as_bytes())?;

        // Show success animation
        println!("✨ Transcription complete! ✨");
        
        // Show the transcription with typing effect
        show_transcription_typing_effect(transcription_text).await;

        println!("\n📁 Transcript saved to: ~/scriba_recordings/{}/transcript.txt", output_path.display());

        // Save transcript to database
        let mut db = Database::new().context("Failed to connect to database")?;
        
        // Find the recording by directory name
        let directory_name = output_path.to_string_lossy().to_string();
        match db.get_recording_by_directory(&directory_name) {
            Ok(Some(recording)) => {
                if let Some(recording_id) = recording.id {
                    // Create transcript record
                    let transcript = Transcript {
                        id: None,
                        recording_id,
                        content: transcription_text.to_string(),
                        created_at: Utc::now(),
                        updated_at: Utc::now(),
                        word_count: None, // Will be calculated by database
                        character_count: None, // Will be calculated by database
                        language_detected: None,
                        confidence_scores: None,
                        segments: None,
                        entities: None,
                        topics: None,
                    };
                    
                    match db.insert_transcript(&transcript) {
                        Ok(transcript_id) => {
                            // Update recording status
                            if let Err(e) = db.update_recording_transcript_status(recording_id, "completed", true) {
                                eprintln!("⚠️ Warning: Failed to update recording status: {}", e);
                            } else {
                                println!("📊 Transcript saved to database with ID: {}", transcript_id);
                            }
                        },
                        Err(e) => eprintln!("⚠️ Warning: Failed to save transcript to database: {}", e),
                    }
                } else {
                    eprintln!("⚠️ Warning: Recording found but has no ID");
                }
            },
            Ok(None) => eprintln!("⚠️ Warning: No recording found in database for directory: {}", directory_name),
            Err(e) => eprintln!("⚠️ Warning: Failed to query recording from database: {}", e),
        }

        Ok(())
    } else {
        println!("❌ Transcription failed!");
        eprintln!("Error: {}", response.status());

        let text = response.text().await.context("Failed to read response body")?;
        eprintln!("Error Body: {text:#?}");
        Err(anyhow::Error::msg("Transcription failed"))
    }
}
