use anyhow::Context;
use anyhow::Result;
use chrono::Local;
use scriba::audio::{AudioFormat, CompressionSettings};
use scriba::config::{LocalModelSize, ScribaConfig, TranscriptionMode};
use scriba::dashboard::Dashboard;
use scriba::record::{calculate_audio_duration, record};
use scriba::transcribe::transcribe_file;
use std::io::{self, Write};
use std::path::PathBuf;
use structopt::StructOpt;

const VERSION: &str = env!("CARGO_PKG_VERSION");
const ASCII_ART: &str = r#"
 ███████  ██████ ██████  ██ ██████   █████  
██      ██      ██   ██ ██ ██   ██ ██   ██ 
███████ ██      ██████  ██ ██████  ███████ 
     ██ ██      ██   ██ ██ ██   ██ ██   ██ 
███████  ██████ ██   ██ ██ ██████  ██   ██ 
                                           
"#;

#[derive(Debug, StructOpt)]
enum Command {
    Record {
        #[structopt(
            short = "n",
            long = "name",
            help = "Optional name/description for the recording (auto-generated if not provided)"
        )]
        name: Option<String>,
        #[structopt(
            short = "s",
            long = "skip-transcription",
            help = "Skip transcription after recording"
        )]
        skip_transcription: bool,
        #[structopt(
            long = "format",
            help = "Audio format (wav, compressed, mp3)",
            default_value = "wav"
        )]
        format: AudioFormat,
        #[structopt(
            long = "sample-rate",
            help = "Sample rate in Hz",
            default_value = "48000"
        )]
        sample_rate: u32,
        #[structopt(long = "bitrate", help = "Bitrate in kbps for compressed formats")]
        bitrate: Option<u32>,
        #[structopt(
            long = "channels",
            help = "Number of channels (1=mono, 2=stereo)",
            default_value = "1"
        )]
        channels: u16,
        #[structopt(
            long = "speech-optimized",
            help = "Use speech-optimized compression settings"
        )]
        speech_optimized: bool,
        #[structopt(long = "local", help = "Force local transcription (overrides config)")]
        force_local: bool,
        #[structopt(
            long = "model",
            help = "Local Whisper model size (tiny|base|small|medium|large|turbo)"
        )]
        model: Option<LocalModelSize>,
        #[structopt(
            long = "api-key",
            help = "OpenAI API key for API-based transcription (overrides config)"
        )]
        api_key: Option<String>,
    },
    Transcribe {
        #[structopt(
            parse(from_os_str),
            help = "Path to existing recording directory name OR external audio file to import"
        )]
        input: PathBuf,
        #[structopt(
            short = "n",
            long = "name",
            help = "Display name for imported files (auto-generated if not provided)"
        )]
        name: Option<String>,
        #[structopt(long = "local", help = "Force local transcription (overrides config)")]
        force_local: bool,
        #[structopt(
            long = "model",
            help = "Local Whisper model size (tiny|base|small|medium|large|turbo)"
        )]
        model: Option<LocalModelSize>,
        #[structopt(
            long = "api-key",
            help = "OpenAI API key for API-based transcription (overrides config)"
        )]
        api_key: Option<String>,
    },
    Config {
        #[structopt(subcommand)]
        cmd: ConfigCommand,
    },
}

#[derive(Debug, StructOpt)]
enum ConfigCommand {
    Show {
        #[structopt(long = "json", help = "Output in JSON format")]
        json: bool,
    },
    SetLocal {
        #[structopt(help = "Model size (tiny|base|small|medium|large|turbo)")]
        model: LocalModelSize,
    },
    SetApi {
        #[structopt(help = "OpenAI API key")]
        api_key: String,
    },
}

#[derive(Debug, StructOpt)]
#[structopt(name = "scriba", about = "A simple CLI tool for recording and transcribing", version = VERSION)]
struct Cli {
    #[structopt(subcommand)]
    command: Option<Command>,
}

fn generate_filename(name: Option<String>) -> String {
    let timestamp = Local::now().format("%Y-%m-%d_%H-%M-%S");
    match name {
        Some(n) => {
            let sanitized = n
                .replace(' ', "-")
                .replace(['/', '\\', ':', '*', '?', '"', '<', '>', '|'], "_");
            format!("{}_{}", timestamp, sanitized)
        }
        None => format!("{}_recording", timestamp),
    }
}

fn resolve_transcription_mode(
    force_local: bool,
    model: Option<LocalModelSize>,
    api_key: Option<String>,
    config: &ScribaConfig,
) -> Result<TranscriptionMode> {
    if force_local {
        let model_size = model.unwrap_or(LocalModelSize::Medium);
        return Ok(TranscriptionMode::Local { model_size });
    }

    if let Some(key) = api_key {
        return Ok(TranscriptionMode::Api { api_key: key });
    }

    if let Some(model_size) = model {
        return Ok(TranscriptionMode::Local { model_size });
    }

    // Use config default
    Ok(config.transcription.clone())
}

async fn import_audio_file(
    input: PathBuf,
    name: Option<String>,
    force_local: bool,
    model: Option<LocalModelSize>,
    api_key: Option<String>,
) -> Result<()> {
    use chrono::{Local, Utc};
    use scriba::database::{Database, Recording};
    use std::fs;

    // Check if input file exists
    if !input.exists() {
        return Err(anyhow::anyhow!("File not found: {}", input.display()));
    }

    // Generate display name
    let display_name = name.unwrap_or_else(|| {
        input
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("imported_audio")
            .to_string()
    });

    println!("📁 Importing audio file: {}", input.display());
    println!("📝 Display name: {}", display_name);

    // Generate unique directory name based on timestamp
    let timestamp = Local::now().format("%Y-%m-%d_%H-%M-%S");
    let directory_name = format!("{}_{}", timestamp, display_name.replace(' ', "_"));

    // Create destination directory
    let base_path = dirs::home_dir()
        .context("Could not find home directory")?
        .join("scriba_recordings");
    let dest_dir = base_path.join(&directory_name);
    fs::create_dir_all(&dest_dir)?;

    // Copy audio file to destination with standard name
    let file_extension = input
        .extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or("wav");
    let dest_file = dest_dir.join(format!("recording.{}", file_extension));

    println!("📂 Copying file to: {}", dest_file.display());
    fs::copy(&input, &dest_file).with_context(|| {
        format!(
            "Failed to copy {} to {}",
            input.display(),
            dest_file.display()
        )
    })?;

    // Insert into database
    let mut db = Database::new()?;
    let audio_format = file_extension.to_string();
    let recording = Recording {
        id: None,
        directory_name: directory_name.clone(),
        display_name: Some(display_name.clone()),
        created_at: Utc::now(),
        updated_at: Utc::now(),
        duration_seconds: {
            // Calculate duration from the copied file
            match calculate_audio_duration(&dest_file, 44100, 2) {
                Ok(duration) => Some(duration),
                Err(_) => None, // Fallback to None if calculation fails
            }
        },
        file_size_bytes: None, // TODO: We could get file size
        audio_format,
        sample_rate: 44100, // Default value
        channels: 2,        // Default value
        has_transcript: false,
        transcript_status: "pending".to_string(),
        language_code: "auto".to_string(),
        model_used: "whisper".to_string(),
        tags: None,
        summary: None,
        key_points: None,
        action_items: None,
        speakers: None,
        sentiment_score: None,
        search_index: None,
        categories: None,
        confidence_score: None,
        audio_path: format!("recording.{}", file_extension),
        transcript_path: None,
    };

    let recording_id = db
        .insert_recording(&recording)
        .with_context(|| "Failed to insert recording into database")?;

    println!("✅ File imported to database with ID: {}", recording_id);

    // Load config and resolve transcription mode
    let config = ScribaConfig::load()?;
    let transcription_mode = resolve_transcription_mode(force_local, model, api_key, &config)?;

    // Start transcription - pass just the directory name, not the full path
    println!("📝 Starting transcription...");
    let directory_path = PathBuf::from(&directory_name);
    transcribe_file(&directory_path, &directory_path, Some(transcription_mode))
        .await
        .with_context(|| "Transcription failed")?;

    println!("🎉 Import and transcription complete!");
    println!("📁 Files saved in: ~/scriba_recordings/{}/", directory_name);

    Ok(())
}

fn print_banner() {
    println!("{}", ASCII_ART);
    println!("╔────────────────────────────────────────────────────────╗");
    println!(
        "║            SCRIBA v{} - AUDIO WORKSTATION           ║",
        VERSION
    );
    println!("║               Recording & Transcription                ║");
    println!("╚────────────────────────────────────────────────────────╝");
    println!();
}

#[allow(dead_code)]
fn print_main_menu() {
    println!("┌────────────────────────────────────────────────────────┐");
    println!("│                    MAIN MENU                           │");
    println!("├────────────────────────────────────────────────────────┤");
    println!("│  [1] Record Audio + Auto-Transcribe                    │");
    println!("│  [2] Record Audio Only                                 │");
    println!("│  [3] Transcribe Existing File                          │");
    println!("│  [D] Recording Library with Statistics                 │");
    println!("│  [4] Exit                                              │");
    println!("└────────────────────────────────────────────────────────┘");
    println!();
}

#[allow(dead_code)]
fn get_user_input(prompt: &str) -> Result<String> {
    print!(">> {}: ", prompt);
    io::stdout().flush()?;
    let mut input = String::new();
    io::stdin().read_line(&mut input)?;
    Ok(input.trim().to_string())
}

#[allow(dead_code)]
fn get_optional_input(prompt: &str) -> Result<Option<String>> {
    let input = get_user_input(prompt)?;
    if input.is_empty() {
        Ok(None)
    } else {
        Ok(Some(input))
    }
}

#[allow(dead_code)]
async fn interactive_mode() -> Result<()> {
    // Load environment variables from .env file
    dotenv::dotenv().ok();

    print_banner();

    loop {
        print_main_menu();

        let choice = get_user_input("Select option (1-4, D for library)")?;

        match choice.as_str() {
            "1" => {
                println!("\n╭─ RECORD + AUTO-TRANSCRIBE ─────────────────────────────╮");
                let name = get_optional_input("Recording name (optional)")?;

                println!("│ Starting recording session...                          │");
                println!("╰────────────────────────────────────────────────────────╯\n");

                let recording_name = generate_filename(name);
                let audio_output = PathBuf::from(&recording_name);

                // Use speech-optimized WAV compression (dynamic device adaptation happens in record function)
                let compression_settings = CompressionSettings::speech_optimized();
                let record_result = record(audio_output.clone(), Some(compression_settings)).await;

                if record_result.is_ok() {
                    let transcript_output = audio_output.clone();
                    println!("\n🎙️ Recording complete! Starting transcription...");

                    let config = ScribaConfig::load().unwrap_or_default();
                    match transcribe_file(
                        &audio_output,
                        &transcript_output,
                        Some(config.transcription),
                    )
                    .await
                    {
                        Ok(()) => {
                            println!("✅ Transcription complete!");
                            println!("📁 Files saved in: ~/scriba_recordings/{}/", recording_name);
                        }
                        Err(err) => {
                            eprintln!("❌ Transcription failed: {err}");
                        }
                    }
                } else if let Err(err) = record_result {
                    eprintln!("❌ Recording failed: {err}");
                }
            }
            "2" => {
                println!("\n╭─ RECORD AUDIO ONLY ────────────────────────────────────╮");
                let name = get_optional_input("Recording name (optional)")?;

                println!("│ Starting recording session...                          │");
                println!("╰────────────────────────────────────────────────────────╯\n");

                let recording_name = generate_filename(name);
                let audio_output = PathBuf::from(&recording_name);

                // Use speech-optimized WAV compression (dynamic device adaptation happens in record function)
                let compression_settings = CompressionSettings::speech_optimized();
                match record(audio_output, Some(compression_settings)).await {
                    Ok(()) => {
                        println!("✅ Recording complete!");
                        println!("📁 File saved in: ~/scriba_recordings/{}/", recording_name);
                    }
                    Err(err) => {
                        eprintln!("❌ Recording failed: {err}");
                    }
                }
            }
            "3" => {
                println!("\n╭─ TRANSCRIBE EXISTING FILE ─────────────────────────────╮");
                let input_path = get_user_input("Path to audio file")?;
                let name = get_optional_input("Transcript name (optional)")?;

                let output = if let Some(n) = name {
                    let timestamp = Local::now().format("%Y-%m-%d_%H-%M-%S");
                    let sanitized = n
                        .replace(' ', "-")
                        .replace(['/', '\\', ':', '*', '?', '"', '<', '>', '|'], "_");
                    PathBuf::from(format!("{}_{}_transcript", timestamp, sanitized))
                } else {
                    let timestamp = Local::now().format("%Y-%m-%d_%H-%M-%S");
                    PathBuf::from(format!("{}_transcript", timestamp))
                };

                println!("│ Starting transcription...                              │");
                println!("╰────────────────────────────────────────────────────────╯\n");

                let config = ScribaConfig::load().unwrap_or_default();
                match transcribe_file(
                    &PathBuf::from(input_path),
                    &output,
                    Some(config.transcription),
                )
                .await
                {
                    Ok(()) => {
                        println!("✅ Transcription complete!");
                        println!(
                            "📁 File saved in: ~/scriba_recordings/{}/",
                            output.display()
                        );
                    }
                    Err(err) => {
                        eprintln!("❌ Transcription failed: {err}");
                    }
                }
            }
            "4" => {
                println!("\n🎵 Thanks for using SCRIBA! Goodbye! 🎵\n");
                break;
            }
            "d" | "D" => {
                println!("\n╭─ RECORDING LIBRARY WITH STATISTICS ────────────────────╮");
                println!("│ Launching enhanced library interface...                │");
                println!("╰────────────────────────────────────────────────────────╯\n");

                match Dashboard::new() {
                    Ok(mut dashboard) => {
                        if let Err(err) = dashboard.run().await {
                            eprintln!("❌ Library error: {err}");
                        }
                    }
                    Err(err) => {
                        eprintln!("❌ Failed to open library: {err}");
                    }
                }
            }
            _ => {
                println!("❌ Invalid choice. Please select 1-4, D.\n");
            }
        }

        if choice != "4" {
            println!("\n{}", "─".repeat(60));
            get_user_input("Press Enter to continue")?;
            println!();
        }
    }

    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::from_args();

    // If no command is provided, launch dashboard directly
    let result = match cli.command {
        None => {
            // Load environment variables from .env file
            dotenv::dotenv().ok();

            // Show ASCII art banner
            print_banner();

            // Launch dashboard directly
            println!("\n╭─ SCRIBA DASHBOARD ─────────────────────────────────────╮");
            println!("│ Launching dashboard interface...                       │");
            println!("╰────────────────────────────────────────────────────────╯\n");

            match Dashboard::new() {
                Ok(mut dashboard) => dashboard.run().await,
                Err(err) => {
                    eprintln!("❌ Failed to open dashboard: {err}");
                    Err(err)
                }
            }
        }
        Some(command) => {
            // Load environment variables for CLI mode
            dotenv::dotenv().ok();

            match command {
                Command::Record {
                    name,
                    skip_transcription,
                    format,
                    sample_rate,
                    bitrate,
                    channels,
                    speech_optimized,
                    force_local,
                    model,
                    api_key,
                } => {
                    // Generate automatic filename
                    let recording_name = generate_filename(name.clone());
                    let audio_output = PathBuf::from(&recording_name);

                    // Create compression settings
                    let compression_settings = CompressionSettings {
                        format,
                        sample_rate,
                        bitrate_kbps: bitrate,
                        channels,
                        speech_optimized,
                    };

                    // Start the recording task
                    let record_result =
                        record(audio_output.clone(), Some(compression_settings)).await;

                    if !skip_transcription {
                        // Use the same directory as the recording
                        let transcript_output = audio_output.clone();

                        // Load config and resolve transcription mode
                        let config = ScribaConfig::load()?;
                        let transcription_mode =
                            resolve_transcription_mode(force_local, model, api_key, &config)?;

                        match transcribe_file(
                            &audio_output,
                            &transcript_output,
                            Some(transcription_mode),
                        )
                        .await
                        {
                            Ok(()) => {
                                println!("Transcription saved to: {:?}", transcript_output);
                            }
                            Err(err) => {
                                eprintln!("Error during transcription: {err}");
                            }
                        }
                    }

                    // Return the result of the record task
                    record_result
                }
                Command::Transcribe {
                    input,
                    name,
                    force_local,
                    model,
                    api_key,
                } => {
                    // Detect if input is an external audio file (import + transcribe) or existing recording directory
                    if input.is_file() && input.extension().is_some() {
                        // External audio file - import and transcribe
                        println!("📁 Detected external audio file, importing and transcribing...");
                        import_audio_file(input, name, force_local, model, api_key).await
                    } else {
                        // Existing recording directory - transcribe only
                        println!("📝 Transcribing existing recording...");

                        // Generate automatic transcript filename if needed
                        let output = if let Some(n) = name {
                            let timestamp = Local::now().format("%Y-%m-%d_%H-%M-%S");
                            let sanitized = n
                                .replace(' ', "-")
                                .replace(['/', '\\', ':', '*', '?', '"', '<', '>', '|'], "_");
                            PathBuf::from(format!("{}_{}_transcript", timestamp, sanitized))
                        } else {
                            let timestamp = Local::now().format("%Y-%m-%d_%H-%M-%S");
                            PathBuf::from(format!("{}_transcript", timestamp))
                        };

                        // Load config and resolve transcription mode
                        let config = ScribaConfig::load()?;
                        let transcription_mode =
                            resolve_transcription_mode(force_local, model, api_key, &config)?;

                        // Transcribe the specified file
                        transcribe_file(&input, &output, Some(transcription_mode)).await
                    }
                }
                Command::Config { cmd } => match cmd {
                    ConfigCommand::Show { json } => {
                        let config = ScribaConfig::load()?;
                        if json {
                            println!("{}", serde_json::to_string_pretty(&config)?);
                        } else {
                            match &config.transcription {
                                TranscriptionMode::Local { model_size } => {
                                    println!("Transcription Mode: Local");
                                    println!("Model Size: {}", model_size);
                                }
                                TranscriptionMode::Api { api_key: _ } => {
                                    println!("Transcription Mode: OpenAI API");
                                    println!("API Key: ***configured***");
                                }
                            }
                            println!("Audio Settings:");
                            println!("  Sample Rate: {} Hz", config.audio_settings.sample_rate);
                            println!("  Bitrate: {} kbps", config.audio_settings.bitrate);
                            println!("  Channels: {}", config.audio_settings.channels);
                            println!(
                                "  Speech Optimized: {}",
                                config.audio_settings.speech_optimized
                            );
                        }
                        Ok(())
                    }
                    ConfigCommand::SetLocal { model } => {
                        let mut config = ScribaConfig::load()?;
                        config.set_transcription_mode(TranscriptionMode::Local {
                            model_size: model,
                        })?;
                        println!(
                            "✅ Updated transcription mode to local with {} model",
                            model
                        );
                        Ok(())
                    }
                    ConfigCommand::SetApi { api_key } => {
                        let mut config = ScribaConfig::load()?;
                        config.set_transcription_mode(TranscriptionMode::Api { api_key })?;
                        println!("✅ Updated transcription mode to OpenAI API");
                        Ok(())
                    }
                },
            }
        }
    };

    if let Err(err) = result {
        eprintln!("An error happened: {err}");
    }

    Ok(())
}
