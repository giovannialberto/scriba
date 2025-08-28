use std::path::{Path, PathBuf};
use anyhow::{Context, Result};
use std::fs::File;
use std::io::{BufWriter, Write};
use lazy_static::lazy_static;
use dirs::home_dir;
use std::time::{Duration, Instant};
use tokio::time::sleep;
use std::io::stdout;
use unicode_width::UnicodeWidthStr;
use crate::database::{Database, Transcript};
use crate::config::{TranscriptionMode, LocalModelSize, ScribaConfig};
use chrono::Utc;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};
use std::ffi::{c_char, c_void};
use std::sync::Once;
extern "C" {
    fn whisper_log_set(
        callback: Option<unsafe extern "C" fn(i32, *const c_char, *mut c_void)>,
        user_data: *mut c_void,
    );
}
use std::process::Command;
use reqwest::{Client, multipart::{Form, Part}};
use futures_util::StreamExt;
use serde_json::Value;

lazy_static! {
    static ref BASE_PATH: PathBuf = home_dir().expect("error home dir").join("scriba_recordings");
}

static INIT_WHISPER_LOG: Once = Once::new();

unsafe extern "C" fn discard_whisper_log(_level: i32, _text: *const c_char, _ud: *mut c_void) {
    // intentionally no-op to keep TUI clean
}

fn ensure_whisper_logs_suppressed() {
    INIT_WHISPER_LOG.call_once(|| unsafe {
        whisper_log_set(Some(discard_whisper_log), std::ptr::null_mut());
    });
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

    async fn show_progress(&mut self, mode_message: Option<&str>) {
        let elapsed = self.start_time.elapsed().as_secs();
        
        // Rotating spinner animation
        let spinner_chars = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
        let spinner = spinner_chars[self.animation_frame % spinner_chars.len()];
        
        // Progress messages that change over time (will be overridden by specific mode)
        let message = match elapsed {
            0..=3 => "Preparing audio",
            4..=8 => "Processing",
            9..=25 => "Transcribing",
            _ => "Almost there, hang tight",
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
        
        let display_message = mode_message.unwrap_or(message);
        print!("\r🎵 {} [{}] {} - {}", spinner, bar_str, display_message, time_display);
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

fn find_ffmpeg() -> Result<String> {
    let possible_paths = [
        "ffmpeg",
        "/opt/homebrew/bin/ffmpeg", // macOS Homebrew
        "/usr/bin/ffmpeg",         // Linux
        "/usr/local/bin/ffmpeg",   // Linux alternative
        "C:\\ffmpeg\\bin\\ffmpeg.exe", // Windows
    ];
    
    for path in &possible_paths {
        match Command::new(path).arg("-version").output() {
            Ok(output) => {
                if output.status.success() {
                    return Ok(path.to_string());
                }
            }
            Err(_) => {
                // Continue to next path on error
                continue;
            }
        }
    }
    
    // If we reach here, none of the standard paths worked
    // Try to use the system PATH by just calling 'ffmpeg'
    match Command::new("ffmpeg").arg("-version").output() {
        Ok(output) if output.status.success() => {
            return Ok("ffmpeg".to_string());
        }
        _ => {}
    }
    
    Err(anyhow::anyhow!(
        "FFmpeg not found. Please install FFmpeg and ensure it's in your PATH. Visit https://ffmpeg.org/download.html"
    ))
}

fn validate_audio_file(path: &Path) -> Result<()> {
    if !path.exists() {
        return Err(anyhow::anyhow!("Audio file not found: {}", path.display()));
    }
    
    let metadata = std::fs::metadata(path)
        .context("Failed to read file metadata")?;
    
    if metadata.len() == 0 {
        return Err(anyhow::anyhow!("Audio file is empty"));
    }
    
    if metadata.len() < 1024 {
        return Err(anyhow::anyhow!("Audio file too small (< 1KB), likely corrupted"));
    }
    
    // Check file extension
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        match ext.to_lowercase().as_str() {
            "wav" | "mp3" | "m4a" | "flac" | "ogg" | "aac" => Ok(()),
            _ => Err(anyhow::anyhow!(
                "Unsupported audio format: {}. Supported: wav, mp3, m4a, flac, ogg, aac", 
                ext
            ))
        }
    } else {
        Err(anyhow::anyhow!("File has no extension, cannot determine audio format"))
    }
}

async fn download_model_with_timeout(model_size: LocalModelSize) -> Result<PathBuf> {
    ensure_model_path_local(model_size, true).await
}

fn resolve_audio_path(input_path: &PathBuf) -> Result<PathBuf> {
    if input_path.is_absolute() {
        if input_path.exists() {
            validate_audio_file(input_path)?;
            Ok(input_path.clone())
        } else {
            Err(anyhow::anyhow!("Audio file not found: {}", input_path.display()))
        }
    } else if input_path.extension().is_some() {
        let full_path = BASE_PATH.join(input_path);
        if full_path.exists() {
            validate_audio_file(&full_path)?;
            Ok(full_path)
        } else {
            Err(anyhow::anyhow!("Audio file not found: {}", full_path.display()))
        }
    } else {
        let recording_dir = BASE_PATH.join(input_path);
        for audio_file in ["recording.mp3", "recording.wav"] {
            let path = recording_dir.join(audio_file);
            if path.exists() {
                validate_audio_file(&path)?;
                return Ok(path);
            }
        }
        Err(anyhow::anyhow!(
            "No audio file found (recording.wav or recording.mp3) in {}", 
            recording_dir.display()
        ))
    }
}

// Silent transcription function for TUI usage (no stdout prints)
pub async fn transcribe_file_silent(input_path: &PathBuf, model: Option<LocalModelSize>) -> Result<(), anyhow::Error> {
    // Suppress any whisper/ggml logs globally to avoid interfering with TUI
    ensure_whisper_logs_suppressed();
    
    let audio_file_path = resolve_audio_path(input_path)?;
    let tmp_wav = ensure_mono_16k_wav(&audio_file_path)?;
    let model_choice = model.unwrap_or(LocalModelSize::Turbo);
    
    // Add timeout wrapper for model operations
    let download_future = download_model_with_timeout(model_choice);
    let model_path = tokio::time::timeout(Duration::from_secs(300), download_future)
        .await
        .context("Model download timed out after 5 minutes")?
        .context("Failed to download model")?;

    let transcript = run_whisper_transcription(&model_path, &tmp_wav)?;

    // Store transcript in same folder as the audio file
    let audio_dir = audio_file_path.parent()
        .context("Could not determine audio file directory")?;
    let transcript_file_path = audio_dir.join("transcript.txt");
    
    let transcript_file = File::create(&transcript_file_path)?;
    let mut transcript_writer = BufWriter::new(transcript_file);
    transcript_writer.write_all(transcript.as_bytes())?;

    // Save transcript to database and link to existing recording
    let mut db = Database::new().context("Failed to connect to database")?;
    let directory_name = audio_dir.file_name()
        .and_then(|name| name.to_str())
        .context("Could not determine directory name")?;
    if let Ok(Some(recording)) = db.get_recording_by_directory(directory_name) {
        if let Some(recording_id) = recording.id {
            let transcript_row = Transcript {
                id: None,
                recording_id,
                content: transcript.clone(),
                created_at: Utc::now(),
                updated_at: Utc::now(),
                word_count: None,
                character_count: None,
                language_detected: None,
                confidence_scores: None,
                segments: None,
                entities: None,
                topics: None,
            };
            if let Ok(_tid) = db.insert_transcript(&transcript_row) {
                let _ = db.update_recording_transcript_status(recording_id, "completed", true);
            }
        }
    }

    Ok(())
}

// Enhanced transcription function with delightful UX and dual mode support
pub async fn transcribe_file(
    input_path: &PathBuf,
    _output_path: &PathBuf,
    mode_override: Option<TranscriptionMode>,
) -> Result<(), anyhow::Error> {
    let audio_file_path = resolve_audio_path(input_path)?;
    
    // Determine which transcription mode to use
    let config = ScribaConfig::load()?;
    let transcription_mode = mode_override.unwrap_or_else(|| config.transcription.clone());
    
    // Create progress indicator
    let progress = TranscriptionProgress::new();
    
    // Show initial progress with mode-specific message
    let mode_description = match &transcription_mode {
        TranscriptionMode::Local { model_size } => {
            format!("🎤 → 📝 Transcribing locally using Whisper {} model...", model_size)
        }
        TranscriptionMode::Api { .. } => {
            "🎤 → ☁️ Transcribing using OpenAI Whisper API...".to_string()
        }
    };
    println!("\n{}\n", mode_description);

    let transcription_text = match transcription_mode {
        TranscriptionMode::Local { model_size } => {
            // Start the progress animation for local mode
            let mut local_progress = progress;
            let progress_task = tokio::spawn(async move {
                loop {
                    let message = match local_progress.start_time.elapsed().as_secs() {
                        0..=3 => Some("Preparing audio (16kHz mono)"),
                        4..=8 => Some("Loading Whisper model"),
                        9..=25 => Some("Running local transcription"),
                        _ => Some("Almost there, hang tight"),
                    };
                    local_progress.show_progress(message).await;
                }
            });
            
            // Prepare audio and model
            let wav_path = ensure_mono_16k_wav(&audio_file_path)
                .context("Failed to prepare 16kHz mono WAV for transcription")?;
                
            let model_path = {
                let download_future = download_model_with_timeout(model_size);
                tokio::time::timeout(Duration::from_secs(300), download_future)
                    .await
                    .context("Model download timed out after 5 minutes")?
                    .context("Failed to download model")?
            };

            // Run local transcription
            let result = run_whisper_transcription(&model_path, &wav_path)
                .context("Local Whisper transcription failed")?;
                
            progress_task.abort();
            result
        }
        TranscriptionMode::Api { api_key } => {
            // Start the progress animation for API mode
            let mut api_progress = progress;
            let progress_task = tokio::spawn(async move {
                loop {
                    let message = match api_progress.start_time.elapsed().as_secs() {
                        0..=3 => Some("Uploading audio file"),
                        4..=8 => Some("OpenAI is processing your audio"),
                        9..=15 => Some("Converting speech to text"),
                        16..=25 => Some("This is taking longer than usual"),
                        _ => Some("Almost there, hang tight"),
                    };
                    api_progress.show_progress(message).await;
                }
            });
            
            let result = transcribe_with_openai_api(&audio_file_path, &api_key).await
                .context("OpenAI API transcription failed")?;
                
            progress_task.abort();
            result
        }
    };

    // Clear progress line
    print!("\r{}", " ".repeat(80));
    print!("\r");
    stdout().flush().unwrap();

    // Show success animation
    println!("✨ Transcription complete! ✨");
    
    // Show the transcription with typing effect
    show_transcription_typing_effect(&transcription_text).await;

    // Store transcript in same folder as the audio file
    let audio_dir = audio_file_path.parent()
        .context("Could not determine audio file directory")?;
    let transcript_file_path = audio_dir.join("transcript.txt");
    
    let transcript_file = File::create(&transcript_file_path)?;
    let mut transcript_writer = BufWriter::new(transcript_file);
    transcript_writer.write_all(transcription_text.as_bytes())?;

    println!("\n📁 Transcript saved to: {}", transcript_file_path.display());

        // Save transcript to database and link to existing recording
        let mut db = Database::new().context("Failed to connect to database")?;
        
        // Find the recording by looking for the directory name that contains this audio file
        let directory_name = audio_dir.file_name()
            .and_then(|name| name.to_str())
            .context("Could not determine directory name")?;
            
        match db.get_recording_by_directory(directory_name) {
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
}

async fn transcribe_with_openai_api(audio_path: &PathBuf, api_key: &str) -> Result<String> {
    let audio_file = std::fs::read(audio_path)
        .context("Unable to read audio file")?;
        
    let filename = audio_path.file_name()
        .and_then(|f| f.to_str())
        .unwrap_or("audio")
        .to_string();
    
    let client = Client::new();
    
    let part = Part::bytes(audio_file)
        .file_name(filename.clone())
        .mime_str("audio/mpeg")
        .context("Failed to create multipart form data")?;

    let form = Form::new()
        .part("file", part)
        .text("model", "whisper-1");

    let response = client
        .post("https://api.openai.com/v1/audio/transcriptions")
        .header("Authorization", format!("Bearer {}", api_key))
        .multipart(form)
        .send()
        .await
        .context("Failed to send transcription request to OpenAI")?;

    let status = response.status();
    if !status.is_success() {
        let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
        return Err(anyhow::anyhow!(
            "OpenAI API request failed with status {}: {}", 
            status, 
            error_text
        ));
    }

    let response_json: Value = response.json().await
        .context("Failed to parse OpenAI response as JSON")?;

    response_json
        .get("text")
        .and_then(|t| t.as_str())
        .map(|s| s.to_string())
        .ok_or_else(|| anyhow::anyhow!("No 'text' field found in OpenAI response"))
}


fn ensure_mono_16k_wav(input: &Path) -> Result<PathBuf> {
    // Always normalize using ffmpeg for simplicity and robustness
    let out = input
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join("_tmp_whisper_16k.wav");

    let ffmpeg_path = find_ffmpeg()
        .context("FFmpeg is required for audio processing")?;

    let output = Command::new(&ffmpeg_path)
        .args([
            "-y",
            "-i",
            input.to_string_lossy().as_ref(),
            "-ar",
            "16000",
            "-ac",
            "1",
            "-f",
            "wav",
            out.to_string_lossy().as_ref(),
        ])
        .output()
        .with_context(|| format!("Failed to run ffmpeg from path: {}", ffmpeg_path))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(anyhow::anyhow!("ffmpeg conversion failed: {}", stderr));
    }

    Ok(out)
}

async fn ensure_model_path_local(size: LocalModelSize, quiet: bool) -> Result<PathBuf> {
    let models_dir = BASE_PATH.join("models");
    std::fs::create_dir_all(&models_dir).ok();

    // Prefer local candidates first (supports both GGML and GGUF file names)
    let local_candidates: Vec<&str> = match size {
        LocalModelSize::Tiny => vec!["ggml-tiny.bin", "tiny.gguf", "ggml-tiny-q5_0.gguf"],
        LocalModelSize::Base => vec!["ggml-base.bin", "base.gguf", "ggml-base-q5_0.gguf"],
        LocalModelSize::Small => vec!["ggml-small.bin", "small.gguf", "ggml-small-q5_0.gguf"],
        LocalModelSize::Medium => vec!["ggml-medium.bin", "medium.gguf", "ggml-medium-q5_0.gguf"],
        LocalModelSize::Large => vec!["ggml-large-v3.bin", "large-v3.gguf", "ggml-large-v3-q5_0.gguf"],
        LocalModelSize::Turbo => vec![
            "ggml-large-v3-turbo.gguf",
            "ggml-large-v3-turbo-q5_0.gguf",
            "large-v3-turbo.gguf",
            "ggml-large-v3-turbo.bin",
        ],
    };

    for name in &local_candidates {
        let p = models_dir.join(name);
        if p.exists() {
            return Ok(p);
        }
    }

    // Remote URL selection (prefer GGUF where appropriate)
    match size {
        LocalModelSize::Tiny => {
            let name = "ggml-tiny.bin";
            let url = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.bin";
            let dest = models_dir.join(name);
            if !quiet { println!("📥 Downloading Whisper model: {} (this may take a while)", name); }
            download_file_streaming(url, &dest, quiet).await
                .with_context(|| format!("Failed to download model from {}", url))?;
            if !quiet { println!("✅ Model downloaded to {}", dest.display()); }
            Ok(dest)
        }
        LocalModelSize::Base => {
            let name = "ggml-base.bin";
            let url = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin";
            let dest = models_dir.join(name);
            if !quiet { println!("📥 Downloading Whisper model: {} (this may take a while)", name); }
            download_file_streaming(url, &dest, quiet).await
                .with_context(|| format!("Failed to download model from {}", url))?;
            if !quiet { println!("✅ Model downloaded to {}", dest.display()); }
            Ok(dest)
        }
        LocalModelSize::Small => {
            let name = "ggml-small.bin";
            let url = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin";
            let dest = models_dir.join(name);
            if !quiet { println!("📥 Downloading Whisper model: {} (this may take a while)", name); }
            download_file_streaming(url, &dest, quiet).await
                .with_context(|| format!("Failed to download model from {}", url))?;
            if !quiet { println!("✅ Model downloaded to {}", dest.display()); }
            Ok(dest)
        }
        LocalModelSize::Medium => {
            let name = "ggml-medium.bin";
            let url = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.bin";
            let dest = models_dir.join(name);
            if !quiet { println!("📥 Downloading Whisper model: {} (this may take a while)", name); }
            download_file_streaming(url, &dest, quiet).await
                .with_context(|| format!("Failed to download model from {}", url))?;
            if !quiet { println!("✅ Model downloaded to {}", dest.display()); }
            Ok(dest)
        }
        LocalModelSize::Large => {
            let name = "ggml-large-v3.bin";
            let url = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3.bin";
            let dest = models_dir.join(name);
            if !quiet { println!("📥 Downloading Whisper model: {} (this may take a while)", name); }
            download_file_streaming(url, &dest, quiet).await
                .with_context(|| format!("Failed to download model from {}", url))?;
            if !quiet { println!("✅ Model downloaded to {}", dest.display()); }
            Ok(dest)
        }
        LocalModelSize::Turbo => {
            // Try a few known turbo filenames/URLs commonly used in the whisper.cpp repo
            let candidates = vec![
                ("ggml-large-v3-turbo-q5_0.gguf", "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo-q5_0.gguf"),
                ("ggml-large-v3-turbo.gguf", "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.gguf"),
                ("large-v3-turbo.gguf", "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/large-v3-turbo.gguf"),
                ("ggml-large-v3-turbo.bin", "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin"),
            ];
            let mut last_err: Option<anyhow::Error> = None;
            for (name, url) in candidates {
                let dest = models_dir.join(name);
                if !quiet { println!("📥 Downloading Whisper model: {} (this may take a while)", name); }
                match download_file_streaming(url, &dest, quiet).await {
                    Ok(()) => {
                        if !quiet { println!("✅ Model downloaded to {}", dest.display()); }
                        return Ok(dest);
                    }
                    Err(e) => {
                        last_err = Some(e);
                        // Try next candidate
                    }
                }
            }
            Err(last_err.unwrap_or_else(|| anyhow::anyhow!("Failed to download turbo model from all known locations")))
        }
    }
}

fn run_whisper_transcription(model_path: &Path, wav_path: &Path) -> Result<String> {
    // Validate inputs
    if !model_path.exists() {
        return Err(anyhow::anyhow!("Whisper model file not found: {}", model_path.display()));
    }
    if !wav_path.exists() {
        return Err(anyhow::anyhow!("Audio file not found: {}", wav_path.display()));
    }
    
    // Load model with enhanced error messages
    let params_ctx = WhisperContextParameters::default();
    let model_path_str = model_path.to_string_lossy();
    let ctx = match WhisperContext::new_with_params(&model_path_str, params_ctx) {
        Ok(ctx) => ctx,
        Err(e) => {
            return Err(anyhow::anyhow!(
                "Failed to load Whisper model '{}': {}. Check if model file is valid and you have enough memory",
                model_path.display(),
                e
            ))
        }
    };
    
    let mut state = ctx.create_state()
        .with_context(|| format!("Failed to create Whisper state for model: {}", model_path.display()))?;

    // Read WAV into f32 PCM
    let mut reader = hound::WavReader::open(wav_path)
        .context("Failed to open normalized WAV for transcription")?;
    let spec = reader.spec();

    // Collect samples as f32 mono
    let mut samples_f32: Vec<f32> = Vec::new();
    match spec.sample_format {
        hound::SampleFormat::Float => {
            for s in reader.samples::<f32>() {
                samples_f32.push(s.unwrap_or(0.0));
            }
        }
        hound::SampleFormat::Int => {
            let max = (1i64 << (spec.bits_per_sample as i64 - 1)) as f32;
            for s in reader.samples::<i32>() {
                let v = s.unwrap_or(0) as f32 / max;
                samples_f32.push(v);
            }
        }
    }

    // Configure params
    let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
    params.set_print_progress(false);
    params.set_print_special(false);
    params.set_print_timestamps(false);
    params.set_single_segment(false);
    params.set_n_threads(num_cpus::get() as i32);

    // Run
    state
        .full(params, &samples_f32)
        .context("Whisper full() failed")?;

    // Collect segments
    let num_segments = state.full_n_segments();
    let mut text = String::new();
    for i in 0..num_segments {
        if let Some(seg) = state.get_segment(i) {
            if let Ok(segment_text) = seg.to_str() {
                if !text.is_empty() {
                    text.push(' ');
                }
                text.push_str(segment_text.trim());
            }
        }
    }

    Ok(text)
}

async fn download_file_streaming(url: &str, dest: &Path, quiet: bool) -> Result<()> {
    let client = Client::new();
    let resp = client.get(url).send().await?.error_for_status()?;
    let total = resp.content_length();
    let mut stream = resp.bytes_stream();
    let mut file = std::fs::File::create(dest)
        .context("Failed to create destination file")?;
    let mut downloaded: u64 = 0;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        std::io::Write::write_all(&mut file, &chunk)?;
        if !quiet {
            downloaded += chunk.len() as u64;
            if let Some(total) = total {
                let pct = (downloaded as f64 / total as f64) * 100.0;
                if downloaded % (10 * 1024 * 1024) < chunk.len() as u64 {
                    print!("\rDownloading model... {:>6.2}%", pct);
                    let _ = stdout().flush();
                }
            }
        }
    }
    if !quiet { println!(); }
    Ok(())
}
