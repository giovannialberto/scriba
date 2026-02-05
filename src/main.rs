use anyhow::Result;
use scriba::core::{
    resolve_transcription_mode, AudioFormat, CompressionSettings, LocalModelSize, ScribaConfig,
    TranscriptionMode, WorkflowManager,
};
use scriba::mcp::run_mcp_server;
use scriba::tui::Dashboard;
use std::path::PathBuf;
use structopt::StructOpt;

const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Print ASCII art with embedded version
fn print_ascii_art() {
    println!(
        r#"
 ███████  ██████ ██████  ██ ██████   █████
██      ██      ██   ██ ██ ██   ██ ██   ██
███████ ██      ██████  ██ ██████  ███████
     ██ ██      ██   ██ ██ ██   ██ ██   ██
███████  ██████ ██   ██ ██ ██████  ██   ██
                                    v{VERSION}
"#
    );
}

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
    Health {
        #[structopt(long = "verbose", help = "Show detailed health information")]
        verbose: bool,
    },
    /// Run the Model Context Protocol (MCP) server over stdio
    Mcp,
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
#[structopt(name = "scriba", about = "A CLI & TUI for recording and transcribing anything", version = VERSION)]
struct Cli {
    #[structopt(subcommand)]
    command: Option<Command>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::from_args();

    // If no command is provided, launch dashboard directly
    let result = match cli.command {
        None => {
            // Load environment variables from .env file
            dotenv::dotenv().ok();

            // Show ASCII art with version
            print_ascii_art();

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
                    // Create compression settings
                    let compression_settings = CompressionSettings {
                        format,
                        sample_rate,
                        bitrate_kbps: bitrate,
                        channels,
                        speech_optimized,
                    };

                    // Load config and resolve transcription mode
                    let config = ScribaConfig::load()?;
                    let transcription_mode = if skip_transcription {
                        None
                    } else {
                        Some(resolve_transcription_mode(
                            force_local,
                            model,
                            api_key,
                            &config,
                        )?)
                    };

                    // Use unified workflow
                    let mut workflow = WorkflowManager::new()?;
                    let _recording = workflow
                        .record_cli(
                            name,
                            Some(compression_settings),
                            !skip_transcription,
                            transcription_mode,
                        )
                        .await?;

                    Ok(())
                }
                Command::Transcribe {
                    input,
                    name,
                    force_local,
                    model,
                    api_key,
                } => {
                    // Load config and resolve transcription mode
                    let config = ScribaConfig::load()?;
                    let transcription_mode =
                        resolve_transcription_mode(force_local, model, api_key, &config)?;

                    let mut workflow = WorkflowManager::new()?;

                    // Detect if input is an external audio file (import + transcribe) or existing recording directory
                    if input.is_file() && input.extension().is_some() {
                        // External audio file - import and transcribe using unified workflow
                        println!("📁 Detected external audio file, importing and transcribing...");
                        let _recording = workflow
                            .complete_import_workflow(&input, name, Some(transcription_mode))
                            .await?;
                        println!("🎉 Import and transcription complete!");
                    } else {
                        // Existing recording directory - re-transcribe using unified workflow
                        println!("📝 Re-transcribing existing recording...");
                        let directory_name = input.to_string_lossy();
                        workflow
                            .retranscribe_recording(&directory_name, transcription_mode)
                            .await?;
                    }

                    Ok(())
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
                Command::Health { verbose } => {
                    let workflow = WorkflowManager::new()?;
                    let health_status = workflow.health_check()?;

                    if verbose {
                        health_status.print_report();
                    } else {
                        if health_status.is_healthy() {
                            println!("✅ Scriba is healthy");
                        } else {
                            println!("❌ Scriba has issues - run with --verbose for details");
                            std::process::exit(1);
                        }
                    }

                    Ok(())
                }
                Command::Mcp => {
                    // Run MCP server on stdio
                    run_mcp_server().await
                }
            }
        }
    };

    if let Err(err) = result {
        eprintln!("An error happened: {err}");
    }

    Ok(())
}
