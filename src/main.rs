use anyhow::Result;
use scriba::core::{
    resolve_transcription_mode, AudioFormat, AutopilotOptions, CloudProvider, CompressionSettings, EndpointPreset, TranscriptionPreset, DEFAULT_OLLAMA_ENDPOINT, DEFAULT_OLLAMA_MODEL,
    EnrichmentMode, LocalModel, RecordingStatus, ScribaConfig, TranscriptionMode, WorkflowManager,
    initialize_world_from_seed, run_autopilot, watcher_excludes_self,
};
use scriba::database::Database;
use scriba::enrichment::WorldContext;
use scriba::entities::EntityRegistry;
use scriba::mcp::run_mcp_server;
use scriba::tui::Dashboard;
use std::io::{self, Read, Write};
use std::path::PathBuf;
use std::sync::Arc;
use structopt::StructOpt;

const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Print ASCII art with embedded version
fn print_ascii_art() {
    let logo = [
        "\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2557} \u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2557}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2557} \u{2588}\u{2588}\u{2557}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2557}  \u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2557} ",
        "\u{2588}\u{2588}\u{2554}\u{2550}\u{2550}\u{2550}\u{2550}\u{255D}\u{2588}\u{2588}\u{2554}\u{2550}\u{2550}\u{2550}\u{2550}\u{255D}\u{2588}\u{2588}\u{2554}\u{2550}\u{2550}\u{2588}\u{2588}\u{2557}\u{2588}\u{2588}\u{2551}\u{2588}\u{2588}\u{2554}\u{2550}\u{2550}\u{2588}\u{2588}\u{2557}\u{2588}\u{2588}\u{2554}\u{2550}\u{2550}\u{2588}\u{2588}\u{2557}",
        "\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2557}\u{2588}\u{2588}\u{2551}     \u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2554}\u{255D}\u{2588}\u{2588}\u{2551}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2554}\u{255D}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2551}",
        "\u{255A}\u{2550}\u{2550}\u{2550}\u{2550}\u{2588}\u{2588}\u{2551}\u{2588}\u{2588}\u{2551}     \u{2588}\u{2588}\u{2554}\u{2550}\u{2550}\u{2588}\u{2588}\u{2557}\u{2588}\u{2588}\u{2551}\u{2588}\u{2588}\u{2554}\u{2550}\u{2550}\u{2588}\u{2588}\u{2557}\u{2588}\u{2588}\u{2554}\u{2550}\u{2550}\u{2588}\u{2588}\u{2551}",
        "\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2551}\u{255A}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2557}\u{2588}\u{2588}\u{2551}  \u{2588}\u{2588}\u{2551}\u{2588}\u{2588}\u{2551}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2554}\u{255D}\u{2588}\u{2588}\u{2551}  \u{2588}\u{2588}\u{2551}",
        "\u{255A}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{255D} \u{255A}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{255D}\u{255A}\u{2550}\u{255D}  \u{255A}\u{2550}\u{255D}\u{255A}\u{2550}\u{255D}\u{255A}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{255D} \u{255A}\u{2550}\u{255D}  \u{255A}\u{2550}\u{255D}",
    ];
    println!();
    for line in &logo {
        println!("  {}", line);
    }
    println!("  v{}\n", VERSION);
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
        #[structopt(
            long = "device",
            help = "Input device name (substring match). Use `scriba health --verbose` to list devices"
        )]
        device: Option<String>,
        #[structopt(
            long = "loopback",
            help = "Enable system audio loopback capture (records both mic and system audio)"
        )]
        loopback: bool,
        #[structopt(
            long = "loopback-device",
            help = "Loopback device name hint (Linux only). On macOS ScreenCaptureKit is used automatically"
        )]
        loopback_device: Option<String>,
        #[structopt(long = "local", help = "Force local transcription (overrides config)")]
        force_local: bool,
        #[structopt(
            long = "model",
            help = "Local model (tiny|base|small|medium|large|turbo|sensevoice|parakeet)"
        )]
        model: Option<LocalModel>,
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
            help = "Local model (tiny|base|small|medium|large|turbo|sensevoice|parakeet)"
        )]
        model: Option<LocalModel>,
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
    /// Run the meeting autopilot in the foreground (the TUI runs it in the
    /// background by default). Detects meetings, asks via a Record/Ignore
    /// panel, records, and stops the moment the meeting app releases the mic.
    ///
    /// Detection works by watching whether a microphone is in use by another
    /// process (e.g. Zoom or Google Meet opening the mic on join) — not by
    /// listening to audio levels — so casual talking never triggers it.
    Watch {
        #[structopt(
            long = "no-auto-record",
            help = "Only fire a desktop notification when a meeting is detected; don't record"
        )]
        no_auto_record: bool,
        #[structopt(
            long = "no-transcribe",
            help = "Skip transcription after an auto-recorded meeting"
        )]
        no_transcribe: bool,
        #[structopt(
            long = "no-confirm",
            help = "Record immediately without asking via the Record/Ignore dialog"
        )]
        no_confirm: bool,
        #[structopt(
            long = "min-silence-seconds",
            help = "Fallback: seconds of silence to auto-stop a meeting recording if the mic-release signal is unavailable"
        )]
        min_silence_seconds: Option<u32>,
        #[structopt(
            long = "device",
            help = "Input device / source name (Linux). On macOS all input devices are monitored"
        )]
        device: Option<String>,
        #[structopt(long = "verbose", help = "Verbose logging from the watcher")]
        verbose: bool,
        #[structopt(
            long = "once",
            help = "Detect and (optionally) record a single meeting, then exit"
        )]
        once: bool,
    },
    /// Run knowledge extraction on an existing recording
    Enrich {
        #[structopt(help = "Recording directory name to enrich")]
        directory_name: String,
        #[structopt(
            long = "enrichment-provider",
            help = "Override enrichment provider (anthropic|openai|google|ollama|custom|deepinfra|openrouter|groq|together)"
        )]
        enrichment_provider: Option<String>,
        #[structopt(
            long = "enrichment-api-key",
            help = "Override enrichment API key"
        )]
        enrichment_api_key: Option<String>,
        #[structopt(
            long = "enrichment-model",
            help = "Override enrichment model"
        )]
        enrichment_model: Option<String>,
        #[structopt(
            long = "enrichment-base-url",
            help = "Override the OpenAI-compatible base URL (with --enrichment-provider custom)"
        )]
        enrichment_base_url: Option<String>,
    },
    /// Teach Scriba voices (owner enrollment, status)
    Voice {
        #[structopt(subcommand)]
        cmd: VoiceCommand,
    },
    /// Manage entities (people, organizations)
    Entity {
        #[structopt(subcommand)]
        cmd: EntityCommand,
    },
    /// Manage Scriba's world context (owner profile)
    World {
        #[structopt(subcommand)]
        cmd: WorldCommand,
    },
    /// Database maintenance commands
    Db {
        #[structopt(subcommand)]
        cmd: DbCommand,
    },
}

#[derive(Debug, StructOpt)]
enum DbCommand {
    /// Rebuild database from recording directories on disk
    Rebuild,
}

#[derive(Debug, StructOpt)]
enum ConfigCommand {
    Show {
        #[structopt(long = "json", help = "Output in JSON format")]
        json: bool,
    },
    SetLocal {
        #[structopt(help = "Model (tiny|base|small|medium|large|turbo|sensevoice|parakeet)")]
        model: LocalModel,
    },
    /// Use a cloud speech API for transcription (OpenAI by default; Groq, DeepInfra or any OpenAI-compatible host)
    SetApi {
        #[structopt(help = "API key for the transcription host")]
        api_key: String,
        #[structopt(long = "preset", help = "Known host: openai|groq|deepinfra (sets endpoint and model)")]
        preset: Option<String>,
        #[structopt(long = "base-url", help = "OpenAI-compatible API root, e.g. https://api.groq.com/openai/v1")]
        base_url: Option<String>,
        #[structopt(long = "model", help = "Transcription model, e.g. whisper-1, gpt-4o-transcribe, whisper-large-v3-turbo")]
        model: Option<String>,
    },
    /// Set the enrichment provider (anthropic, openai, google, ollama, or an OpenAI-compatible host)
    SetProvider {
        #[structopt(help = "Provider name (anthropic|openai|google|ollama|custom|deepinfra|openrouter|groq|together)")]
        provider: String,
        #[structopt(long = "base-url", help = "Base URL for an OpenAI-compatible endpoint (required with 'custom')")]
        base_url: Option<String>,
    },
    /// Set the enrichment API key (for cloud providers)
    SetEnrichmentKey {
        #[structopt(help = "API key")]
        key: String,
    },
    /// Set the enrichment model
    SetEnrichmentModel {
        #[structopt(help = "Model name")]
        model: String,
    },
}

#[derive(Debug, StructOpt)]
enum VoiceCommand {
    /// Learn the owner's voice from an audio file of the owner speaking alone (10-30 s is plenty)
    Enroll {
        #[structopt(help = "Audio file (wav, mp3, m4a...) with only the owner speaking")]
        file: PathBuf,
    },
    /// Show how much of the owner's voice Scriba has learned
    Status,
    /// Forget everything learned about the owner's voice
    Forget,
}

#[derive(Debug, StructOpt)]
enum EntityCommand {
    /// List all entities
    List {
        #[structopt(long = "type", help = "Filter by type (person, organization)")]
        entity_type: Option<String>,
        #[structopt(long = "limit", help = "Limit number of results")]
        limit: Option<i64>,
    },
    /// Show details of a specific entity
    Show {
        #[structopt(help = "Entity ID or name")]
        id_or_name: String,
    },
    /// Rename an entity (old name becomes alias)
    Rename {
        #[structopt(help = "Entity ID")]
        id: i64,
        #[structopt(help = "New name")]
        new_name: String,
    },
    /// Update entity context
    Update {
        #[structopt(help = "Entity ID")]
        id: i64,
        #[structopt(long = "context", help = "New context description")]
        context: String,
    },
    /// Manage entity aliases
    Alias {
        #[structopt(subcommand)]
        cmd: AliasCommand,
    },
    /// Delete an entity
    Delete {
        #[structopt(help = "Entity ID")]
        id: i64,
    },
    /// Merge two entities (source into target)
    Merge {
        #[structopt(help = "Source entity ID (will be deleted)")]
        source_id: i64,
        #[structopt(help = "Target entity ID (will receive merged data)")]
        target_id: i64,
    },
}

#[derive(Debug, StructOpt)]
enum AliasCommand {
    /// Add an alias to an entity
    Add {
        #[structopt(help = "Entity ID")]
        id: i64,
        #[structopt(help = "Alias to add")]
        alias: String,
    },
    /// Remove an alias from an entity
    Remove {
        #[structopt(help = "Entity ID")]
        id: i64,
        #[structopt(help = "Alias to remove")]
        alias: String,
    },
}

#[derive(Debug, StructOpt)]
enum WorldCommand {
    /// Show the current world description
    Show,
    /// Initialize the world with seed content
    Init {
        #[structopt(long = "stdin", help = "Read seed content from stdin")]
        stdin: bool,
    },
    /// Edit the world description (opens in $EDITOR)
    Edit,
    /// Get the path to the world file
    Path,
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
                    device,
                    loopback,
                    loopback_device,
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
                    let mut config = ScribaConfig::load()?;
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

                    // Apply CLI device override
                    if device.is_some() {
                        config.audio_settings.input_device = device;
                    }

                    // Apply CLI loopback override
                    if loopback {
                        // --loopback flag enables it; --loopback-device sets a specific hint
                        config.audio_settings.loopback_device =
                            Some(loopback_device.unwrap_or_default());
                    } else if loopback_device.is_some() {
                        config.audio_settings.loopback_device = loopback_device;
                    }

                    // Use unified workflow
                    let mut workflow = WorkflowManager::with_config(config)?;
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
                                TranscriptionMode::Local { model } => {
                                    println!("Transcription Mode: Local");
                                    println!("Model: {}", model.display_name());
                                }
                                TranscriptionMode::Api { .. } => {
                                    println!("Transcription Mode: Cloud API");
                                    println!("Host: {}", config.transcription_host_display());
                                    println!("Endpoint: {}", config.transcription_base_url());
                                    println!("Model: {}", config.transcription_model());
                                    let key_status = if config.resolve_transcription_api_key().is_some() {
                                        "***configured***"
                                    } else {
                                        "(not set)"
                                    };
                                    println!("API Key: {}", key_status);
                                }
                            }
                            println!("\nEnrichment:");
                            println!("  Enabled: {}", config.enrichment.enabled);
                            println!("  Provider: {}", config.enrichment.provider_display_name());
                            if let Some(url) = config.enrichment.effective_base_url() {
                                println!("  Endpoint: {}", url);
                            }
                            println!("  Model: {}", config.enrichment.model_name());
                            if config.enrichment.needs_api_key() {
                                let key_status = if config.enrichment.resolve_api_key().is_some() {
                                    "***configured***"
                                } else {
                                    "(not set)"
                                };
                                println!("  API Key: {}", key_status);
                            }
                            println!("\nAudio Settings:");
                            println!("  Sample Rate: {} Hz", config.audio_settings.sample_rate);
                            println!("  Bitrate: {} kbps", config.audio_settings.bitrate);
                            println!("  Channels: {}", config.audio_settings.channels);
                            println!(
                                "  Speech Optimized: {}",
                                config.audio_settings.speech_optimized
                            );
                            println!(
                                "  Input Device: {}",
                                config.audio_settings.input_device.as_deref().unwrap_or("(system default)")
                            );
                            println!(
                                "  Loopback: {}",
                                match &config.audio_settings.loopback_device {
                                    Some(d) if d.is_empty() => "enabled (auto-detect)".to_string(),
                                    Some(d) => format!("enabled ({})", d),
                                    None => "disabled".to_string(),
                                }
                            );
                        }
                        Ok(())
                    }
                    ConfigCommand::SetLocal { model } => {
                        let mut config = ScribaConfig::load()?;
                        config.set_transcription_mode(TranscriptionMode::Local {
                            model,
                        })?;
                        println!(
                            "Updated transcription mode to local with {} model",
                            model.display_name()
                        );
                        Ok(())
                    }
                    ConfigCommand::SetApi { api_key, preset, base_url, model } => {
                        let mut config = ScribaConfig::load()?;
                        let (mut new_base_url, mut new_model) = match &config.transcription {
                            TranscriptionMode::Api { base_url, model, .. } => (base_url.clone(), model.clone()),
                            TranscriptionMode::Local { .. } => (
                                config.last_transcription_base_url.clone(),
                                config.last_transcription_model.clone(),
                            ),
                        };
                        if let Some(name) = preset {
                            let p = TranscriptionPreset::by_name(&name).ok_or_else(|| {
                                anyhow::anyhow!(
                                    "Unknown transcription preset '{}'. Use one of: {}",
                                    name,
                                    TranscriptionPreset::names().join(", ")
                                )
                            })?;
                            new_base_url = Some(p.base_url.to_string());
                            new_model = Some(p.model.to_string());
                        }
                        if let Some(url) = base_url {
                            new_base_url = Some(url.trim().trim_end_matches('/').to_string()).filter(|u| !u.is_empty());
                        }
                        if let Some(m) = model {
                            new_model = Some(m.trim().to_string()).filter(|m| !m.is_empty());
                        }
                        config.set_transcription_mode(TranscriptionMode::Api {
                            api_key,
                            base_url: new_base_url,
                            model: new_model,
                        })?;
                        println!(
                            "✅ Cloud transcription: {} ({})",
                            config.transcription_host_display(),
                            config.transcription_model()
                        );
                        Ok(())
                    }
                    ConfigCommand::SetProvider { provider, base_url } => {
                        let mut config = ScribaConfig::load()?;
                        if provider.to_lowercase() == "ollama" {
                            config.enrichment.mode = EnrichmentMode::Local {
                                ollama_endpoint: config.enrichment.last_ollama_endpoint.clone().unwrap_or_else(|| DEFAULT_OLLAMA_ENDPOINT.to_string()),
                                ollama_model: config.enrichment.last_ollama_model.clone().unwrap_or_else(|| DEFAULT_OLLAMA_MODEL.to_string()),
                            };
                        } else {
                            let cloud_provider: CloudProvider = provider.parse()?;
                            let base_url = resolve_base_url(&provider, &cloud_provider, base_url.as_deref())?;
                            let existing_key = config.enrichment.load_key_for_provider(&cloud_provider);
                            let model = config.enrichment.load_model_for_provider(&cloud_provider);
                            config.enrichment.mode = EnrichmentMode::Cloud {
                                provider: cloud_provider.clone(),
                                api_key: existing_key,
                                model,
                                base_url: base_url.clone(),
                            };
                            config.enrichment.save_base_url_for_provider(&cloud_provider, &base_url);
                        }
                        config.save()?;
                        println!("✅ Updated enrichment provider to {}", config.enrichment.provider_display_name());
                        Ok(())
                    }
                    ConfigCommand::SetEnrichmentKey { key } => {
                        let mut config = ScribaConfig::load()?;
                        match &mut config.enrichment.mode {
                            EnrichmentMode::Cloud { api_key, .. } => {
                                *api_key = key;
                            }
                            EnrichmentMode::Local { .. } => {
                                return Err(anyhow::anyhow!(
                                    "Cannot set API key for local (Ollama) mode. Switch to a cloud provider first with: scriba config set-provider <anthropic|openai|google|custom|deepinfra|...>"
                                ));
                            }
                        }
                        config.save()?;
                        println!("✅ Updated enrichment API key");
                        Ok(())
                    }
                    ConfigCommand::SetEnrichmentModel { model } => {
                        let mut config = ScribaConfig::load()?;
                        match &mut config.enrichment.mode {
                            EnrichmentMode::Cloud { model: m, .. } => {
                                *m = Some(model.clone());
                            }
                            EnrichmentMode::Local { ollama_model, .. } => {
                                *ollama_model = model.clone();
                            }
                        }
                        config.save()?;
                        println!("✅ Updated enrichment model to '{}'", model);
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
                Command::Watch {
                    no_auto_record,
                    no_transcribe,
                    no_confirm,
                    min_silence_seconds,
                    device,
                    verbose,
                    once,
                } => {
                    run_watch(
                        no_auto_record,
                        no_transcribe,
                        no_confirm,
                        min_silence_seconds,
                        device,
                        verbose,
                        once,
                    )
                    .await
                }
                Command::Enrich { directory_name, enrichment_provider, enrichment_api_key, enrichment_model, enrichment_base_url } => {
                    println!("🧠 Running knowledge extraction on: {}", directory_name);

                    let mut workflow = if enrichment_provider.is_some() || enrichment_api_key.is_some() || enrichment_model.is_some() {
                        // Apply CLI overrides to config
                        let mut config = ScribaConfig::load()?;
                        apply_enrichment_overrides(&mut config, enrichment_provider.as_deref(), enrichment_api_key.as_deref(), enrichment_model.as_deref(), enrichment_base_url.as_deref())?;
                        WorkflowManager::with_config(config)?
                    } else {
                        WorkflowManager::new()?
                    };

                    workflow.enrich_existing_recording(&directory_name, true).await?;
                    Ok(())
                }
                Command::Voice { cmd } => {
                    let mut db = Database::new()?;
                    match cmd {
                        VoiceCommand::Enroll { file } => {
                            let samples = scriba::core::voice::enroll_owner_from_file(&file, &mut db).await?;
                            let (count, secs) = db.speaker_sample_stats("owner", scriba::core::diarization::EMBEDDING_MODEL_ID)?;
                            println!(
                                "✅ Learned {} voice sample(s) from {}. Total: {} sample(s), {:.0}s of your voice.",
                                samples,
                                file.display(),
                                count,
                                secs
                            );
                        }
                        VoiceCommand::Status => {
                            let (count, secs) = db.speaker_sample_stats("owner", scriba::core::diarization::EMBEDDING_MODEL_ID)?;
                            if count == 0 {
                                println!("Scriba has not learned your voice yet. Run: scriba voice enroll <file>");
                            } else {
                                println!("Owner voice: {} sample(s), {:.0}s of speech.", count, secs);
                            }
                            for p in db.speaker_profiles(scriba::core::diarization::EMBEDDING_MODEL_ID)?.into_iter().filter(|p| !p.is_owner) {
                                println!("{}: {} sample(s), {:.0}s", p.speaker, p.samples, p.total_secs);
                            }
                        }
                        VoiceCommand::Forget => {
                            let n = db.delete_speaker_samples("owner")?;
                            println!("🗑  Forgot {} owner voice sample(s).", n);
                        }
                    }
                    Ok(())
                }
                Command::Entity { cmd } => {
                    let mut db = Database::new()?;
                    let mut registry = EntityRegistry::new(&mut db);

                    match cmd {
                        EntityCommand::List { entity_type, limit } => {
                            let entities =
                                registry.list_entities(entity_type.as_deref(), limit)?;
                            if entities.is_empty() {
                                println!("No entities found.");
                            } else {
                                println!(
                                    "\n{:<4} {:<12} {:<20} {:<30} {:<8}",
                                    "ID", "Type", "Name", "Aliases", "Mentions"
                                );
                                println!("{}", "-".repeat(80));
                                for entity in entities {
                                    let aliases = entity.aliases_list().join(", ");
                                    let aliases_display = if aliases.len() > 28 {
                                        format!("{}...", &aliases[..25])
                                    } else if aliases.is_empty() {
                                        "-".to_string()
                                    } else {
                                        aliases
                                    };
                                    println!(
                                        "{:<4} {:<12} {:<20} {:<30} {:<8}",
                                        entity.id.unwrap_or(0),
                                        entity.entity_type,
                                        entity.canonical_name,
                                        aliases_display,
                                        entity.mention_count
                                    );
                                }
                            }
                            Ok(())
                        }
                        EntityCommand::Show { id_or_name } => {
                            let entity = if let Ok(id) = id_or_name.parse::<i64>() {
                                registry.get_entity(id)?
                            } else {
                                registry.get_entity_by_name(&id_or_name)?
                            };

                            if let Some(entity) = entity {
                                println!("\n╭─ ENTITY ─────────────────────────────────────────╮");
                                println!("│ ID:       {:<40}│", entity.id.unwrap_or(0));
                                println!("│ Type:     {:<40}│", entity.entity_type);
                                println!("│ Name:     {:<40}│", entity.canonical_name);
                                let aliases = entity.aliases_list().join(", ");
                                let aliases_display =
                                    if aliases.is_empty() { "-".to_string() } else { aliases };
                                println!("│ Aliases:  {:<40}│", aliases_display);
                                println!("│ Mentions: {:<40}│", entity.mention_count);
                                println!("├──────────────────────────────────────────────────┤");
                                if let Some(ctx) = &entity.context {
                                    println!("│ Context:                                         │");
                                    // Word-wrap context to fit
                                    for line in ctx.chars().collect::<Vec<_>>().chunks(48) {
                                        let s: String = line.iter().collect();
                                        println!("│   {:<47}│", s);
                                    }
                                } else {
                                    println!("│ Context:  (none)                                 │");
                                }
                                println!("╰──────────────────────────────────────────────────╯");
                            } else {
                                println!("Entity not found: {}", id_or_name);
                            }
                            Ok(())
                        }
                        EntityCommand::Rename { id, new_name } => {
                            if let Some(entity) = registry.get_entity(id)? {
                                let old_name = entity.canonical_name.clone();
                                registry.rename_entity(id, &new_name)?;
                                println!(
                                    "✅ Renamed entity {} from '{}' to '{}'",
                                    id, old_name, new_name
                                );
                                println!("   (old name '{}' added as alias)", old_name);
                            } else {
                                println!("Entity not found: {}", id);
                            }
                            Ok(())
                        }
                        EntityCommand::Update { id, context } => {
                            if registry.get_entity(id)?.is_some() {
                                registry.update_entity_context(id, &context)?;
                                println!("✅ Updated context for entity {}", id);
                            } else {
                                println!("Entity not found: {}", id);
                            }
                            Ok(())
                        }
                        EntityCommand::Alias { cmd: alias_cmd } => match alias_cmd {
                            AliasCommand::Add { id, alias } => {
                                if registry.get_entity(id)?.is_some() {
                                    registry.add_entity_alias(id, &alias)?;
                                    println!("✅ Added alias '{}' to entity {}", alias, id);
                                } else {
                                    println!("Entity not found: {}", id);
                                }
                                Ok(())
                            }
                            AliasCommand::Remove { id, alias } => {
                                if registry.get_entity(id)?.is_some() {
                                    registry.remove_entity_alias(id, &alias)?;
                                    println!("✅ Removed alias '{}' from entity {}", alias, id);
                                } else {
                                    println!("Entity not found: {}", id);
                                }
                                Ok(())
                            }
                        },
                        EntityCommand::Delete { id } => {
                            if let Some(entity) = registry.get_entity(id)? {
                                registry.delete_entity(id)?;
                                println!(
                                    "✅ Deleted entity {}: {}",
                                    id, entity.canonical_name
                                );
                            } else {
                                println!("Entity not found: {}", id);
                            }
                            Ok(())
                        }
                        EntityCommand::Merge {
                            source_id,
                            target_id,
                        } => {
                            let source = registry.get_entity(source_id)?;
                            let target = registry.get_entity(target_id)?;

                            match (source, target) {
                                (Some(src), Some(tgt)) => {
                                    registry.merge_entities(source_id, target_id)?;
                                    println!(
                                        "✅ Merged '{}' into '{}'",
                                        src.canonical_name, tgt.canonical_name
                                    );
                                    println!(
                                        "   '{}' added as alias, mentions transferred",
                                        src.canonical_name
                                    );
                                }
                                (None, _) => println!("Source entity not found: {}", source_id),
                                (_, None) => println!("Target entity not found: {}", target_id),
                            }
                            Ok(())
                        }
                    }
                }
                Command::Db { cmd } => match cmd {
                    DbCommand::Rebuild => {
                        use scriba::core::FileManager;
                        use scriba::utils::BASE_PATH;

                        println!("Rebuilding database from recording directories...\n");
                        let mut db = Database::new()?;
                        let base = BASE_PATH.as_path();

                        let mut rebuilt = 0u32;
                        let mut transcripts_found = 0u32;

                        let mut entries: Vec<_> = std::fs::read_dir(base)?
                            .filter_map(|e| e.ok())
                            .filter(|e| e.path().is_dir())
                            .collect();
                        entries.sort_by_key(|e| e.file_name());

                        for entry in &entries {
                            let dir_path = entry.path();
                            let dir_name = entry.file_name().to_string_lossy().to_string();

                            // Skip if already in DB
                            if db.get_recording_by_directory(&dir_name)?.is_some() {
                                continue;
                            }

                            // Find audio file
                            let audio_path = match FileManager::find_audio_file(&dir_path) {
                                Some(p) => p,
                                None => continue, // not a recording directory
                            };
                            let audio_filename = audio_path
                                .file_name()
                                .unwrap_or_default()
                                .to_string_lossy()
                                .to_string();

                            // Extract metadata
                            let meta = FileManager::extract_audio_metadata(&audio_path)
                                .unwrap_or(scriba::core::RecordingMetadata {
                                    duration_seconds: None,
                                    file_size_bytes: None,
                                    audio_format: "wav".to_string(),
                                    sample_rate: 48000,
                                    channels: 1,
                                });

                            // Parse timestamp from dir name (format: YYYY-MM-DD_HH-MM-SS_*)
                            let created_at = chrono::NaiveDateTime::parse_from_str(
                                &dir_name[..19],
                                "%Y-%m-%d_%H-%M-%S",
                            )
                            .map(|dt| chrono::DateTime::<chrono::Utc>::from_naive_utc_and_offset(dt, chrono::Utc))
                            .unwrap_or_else(|_| chrono::Utc::now());

                            // Display name from dir suffix
                            let display_name = if dir_name.len() > 20 {
                                Some(dir_name[20..].replace('_', " "))
                            } else {
                                None
                            };

                            // Check for transcript
                            let transcript_path = dir_path.join("transcript.txt");
                            let has_transcript = transcript_path.exists();

                            let recording = scriba::database::Recording {
                                id: None,
                                directory_name: dir_name.clone(),
                                display_name,
                                created_at,
                                updated_at: created_at,
                                duration_seconds: meta.duration_seconds,
                                file_size_bytes: meta.file_size_bytes,
                                audio_format: meta.audio_format,
                                sample_rate: meta.sample_rate,
                                channels: meta.channels,
                                has_transcript,
                                transcript_status: if has_transcript {
                                    "completed".to_string()
                                } else {
                                    "pending".to_string()
                                },
                                language_code: "auto".to_string(),
                                model_used: "whisper-1".to_string(),
                                tags: None,
                                summary: None,
                                key_points: None,
                                action_items: None,
                                speakers: None,
                                sentiment_score: None,
                                search_index: None,
                                categories: None,
                                confidence_score: None,
                                audio_path: audio_filename,
                                transcript_error: None,
                                transcript_path: if has_transcript {
                                    Some("transcript.txt".to_string())
                                } else {
                                    None
                                },
                            };

                            let rec_id = db.insert_recording(&recording)?;
                            rebuilt += 1;
                            print!("  + {}", dir_name);

                            if has_transcript {
                                let content = std::fs::read_to_string(&transcript_path)?;
                                db.upsert_transcript(rec_id, &content)?;
                                transcripts_found += 1;
                                println!(" (with transcript)");
                            } else {
                                println!();
                            }
                        }

                        if rebuilt == 0 {
                            println!("Database is already up to date — no missing recordings found.");
                        } else {
                            println!(
                                "\nRebuilt {} recording(s), {} with transcript(s).",
                                rebuilt, transcripts_found
                            );
                        }
                        Ok(())
                    }
                },
                Command::World { cmd } => match cmd {
                    WorldCommand::Show => {
                        let world = WorldContext::load()?;
                        if world.has_content() {
                            println!("\n📍 {}\n", world.path.display());
                            if let Some(data) = world.parsed() {
                                // Structured display
                                println!("Owner: {} ({})", data.owner.name, data.owner.role);
                                if !data.owner.organization.is_empty() {
                                    println!("Organization: {}", data.owner.organization);
                                }
                                if !data.owner.location.is_empty() {
                                    println!("Location: {}", data.owner.location);
                                }
                                if !data.people.is_empty() {
                                    println!("\nPeople:");
                                    for p in &data.people {
                                        if p.relationship.is_empty() {
                                            println!("  - {}", p.name);
                                        } else {
                                            println!("  - {} ({})", p.name, p.relationship);
                                        }
                                    }
                                }
                                if !data.organizations.is_empty() {
                                    println!("\nOrganizations:");
                                    for o in &data.organizations {
                                        if o.description.is_empty() {
                                            println!("  - {}", o.name);
                                        } else {
                                            println!("  - {} — {}", o.name, o.description);
                                        }
                                    }
                                }
                                if !data.interests.is_empty() {
                                    println!("\nInterests: {}", data.interests.join(", "));
                                }
                                if !data.projects.is_empty() {
                                    println!("\nProjects:");
                                    for p in &data.projects {
                                        if p.description.is_empty() {
                                            println!("  - {}", p.name);
                                        } else {
                                            println!("  - {} — {}", p.name, p.description);
                                        }
                                    }
                                }
                                if !data.beliefs.is_empty() {
                                    println!("\nBeliefs:");
                                    for b in &data.beliefs {
                                        println!("  - {}", b);
                                    }
                                }
                            } else {
                                // Legacy or raw content
                                println!("{}", world.content);
                            }
                        } else {
                            println!("🌍 No world context configured yet.");
                            println!("\nTo initialize your world, run:");
                            println!("  scriba world init");
                            println!("\nOr create the file directly at:");
                            println!("  {}", WorldContext::file_path().display());
                        }
                        Ok(())
                    }
                    WorldCommand::Init { stdin } => {
                        if WorldContext::exists() {
                            println!("⚠️ World file already exists at:");
                            println!("   {}", WorldContext::file_path().display());
                            println!("\nUse 'scriba world edit' to modify it.");
                            return Ok(());
                        }

                        let seed_content = if stdin {
                            println!("Reading world seed from stdin...");
                            let mut content = String::new();
                            io::stdin().read_to_string(&mut content)?;
                            content
                        } else {
                            println!("🌍 Initialize Scriba's World\n");
                            println!("Tell Scriba about yourself. This context helps with:");
                            println!("  • Better entity recognition (company names, people)");
                            println!("  • More accurate summaries and titles");
                            println!("  • Understanding your conversations\n");
                            println!("Example:");
                            println!("  I'm Giovanni, co-founder of Exein, a cybersecurity startup.");
                            println!("  Variations like 'Exane', 'Xane' in transcripts refer to Exein.");
                            println!("  I work closely with Luca (CTO) and Gianni (co-founder).\n");
                            print!("Enter your world description (press Enter twice to finish):\n> ");
                            io::stdout().flush()?;

                            let mut lines = Vec::new();
                            let stdin = io::stdin();
                            loop {
                                let mut line = String::new();
                                stdin.read_line(&mut line)?;
                                if line.trim().is_empty() {
                                    break;
                                }
                                lines.push(line);
                                print!("> ");
                                io::stdout().flush()?;
                            }
                            lines.join("")
                        };

                        if seed_content.trim().is_empty() {
                            println!("❌ No content provided. World not initialized.");
                            return Ok(());
                        }

                        println!("\n🔍 Building structured world profile...");
                        let config = ScribaConfig::load()?;
                        let mut db = Database::new()?;

                        match initialize_world_from_seed(&mut db, &config, &seed_content).await? {
                            Some((_world_data, extraction)) => {
                                println!("   ✅ Structured profile created");
                                let entity_count = extraction.people.len() + extraction.organizations.len();
                                for person in &extraction.people {
                                    println!("   ✅ Created person: {} {}",
                                        person.name,
                                        if person.is_owner { "(owner)" } else { "" }
                                    );
                                }
                                for org in &extraction.organizations {
                                    println!("   ✅ Created organization: {}", org.name);
                                }
                                println!("\n🎉 Created {} entities from your world description.", entity_count);
                            }
                            None => {
                                println!("   ⚠️ Enrichment provider not available - saved raw seed text.");
                                println!("   Check your enrichment configuration with: scriba config show");
                            }
                        }

                        println!("\n✅ World initialized at:");
                        println!("   {}", WorldContext::file_path().display());
                        println!("\nScriba will now use this context for all extractions.");
                        println!("The world will evolve automatically as you add recordings.");
                        Ok(())
                    }
                    WorldCommand::Edit => {
                        let path = WorldContext::file_path();

                        // Create file with template if it doesn't exist
                        if !path.exists() {
                            let template = "# Scriba's World\n\n\
                                Tell Scriba about yourself here. This helps with better\n\
                                entity recognition and understanding your conversations.\n\n\
                                Example:\n\
                                I'm [Your Name], [your role] at [your company].\n\
                                [Add context about people you work with, projects, etc.]\n";
                            std::fs::write(&path, template)?;
                        }

                        // Open in editor
                        let editor = std::env::var("EDITOR").unwrap_or_else(|_| "nano".to_string());
                        let status = std::process::Command::new(&editor)
                            .arg(&path)
                            .status()?;

                        if status.success() {
                            println!("✅ World file saved.");
                        } else {
                            println!("⚠️ Editor exited with non-zero status.");
                        }
                        Ok(())
                    }
                    WorldCommand::Path => {
                        println!("{}", WorldContext::file_path().display());
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

/// Resolve the base URL for a cloud provider chosen on the CLI: an explicit
/// `--base-url` wins, then a preset named by the provider string, then none.
fn resolve_base_url(
    provider_str: &str,
    provider: &CloudProvider,
    explicit: Option<&str>,
) -> Result<Option<String>> {
    if let Some(url) = explicit.map(str::trim).filter(|u| !u.is_empty()) {
        return Ok(Some(url.trim_end_matches('/').to_string()));
    }
    if let Some(preset) = EndpointPreset::by_name(provider_str) {
        return Ok(Some(preset.base_url.to_string()));
    }
    if provider.uses_custom_endpoint() {
        return Err(anyhow::anyhow!(
            "Provider 'custom' needs --base-url <URL>, or name a known host: {}",
            EndpointPreset::names().join(", ")
        ));
    }
    Ok(None)
}

/// Apply CLI enrichment overrides to config (without persisting).
fn apply_enrichment_overrides(
    config: &mut ScribaConfig,
    provider: Option<&str>,
    api_key: Option<&str>,
    model: Option<&str>,
    base_url: Option<&str>,
) -> Result<()> {
    if let Some(provider_str) = provider {
        if provider_str.to_lowercase() == "ollama" {
            config.enrichment.mode = EnrichmentMode::Local {
                ollama_endpoint: config.enrichment.ollama_endpoint(),
                ollama_model: model.map(str::to_string).unwrap_or_else(|| config.enrichment.ollama_model()),
            };
            return Ok(());
        }

        let cloud_provider: CloudProvider = provider_str.parse()?;
        let base_url = resolve_base_url(provider_str, &cloud_provider, base_url)?;
        let key = api_key
            .map(|k| k.to_string())
            .or_else(|| Some(config.enrichment.load_key_for_provider(&cloud_provider)))
            .filter(|k| !k.is_empty())
            .unwrap_or_default();
        config.enrichment.mode = EnrichmentMode::Cloud {
            provider: cloud_provider,
            api_key: key,
            model: model.map(|m| m.to_string()),
            base_url,
        };
    } else {
        // No provider override, but possibly api_key or model override
        if let Some(key) = api_key {
            if let EnrichmentMode::Cloud { api_key: k, .. } = &mut config.enrichment.mode {
                *k = key.to_string();
            }
        }
        if let Some(m) = model {
            match &mut config.enrichment.mode {
                EnrichmentMode::Cloud { model: mm, .. } => {
                    *mm = Some(m.to_string());
                }
                EnrichmentMode::Local { ollama_model, .. } => {
                    *ollama_model = m.to_string();
                }
            }
        }
    }
    Ok(())
}

/// CLI wrapper for the meeting autopilot (`scriba watch`): prints the banner,
/// maps Ctrl+C to the shutdown signal, then runs the shared autopilot loop
/// (`core::autopilot`). The same loop runs quietly inside the TUI dashboard
/// by default.
#[allow(clippy::too_many_arguments)]
async fn run_watch(
    no_auto_record: bool,
    no_transcribe: bool,
    no_confirm: bool,
    min_silence_seconds: Option<u32>,
    device: Option<String>,
    verbose: bool,
    once: bool,
) -> Result<()> {
    let mut config = ScribaConfig::load()?;

    // Apply CLI overrides to the meeting detection config (not persisted).
    let md = &mut config.meeting_detection;
    if no_auto_record {
        md.auto_record = false;
    }
    if no_confirm {
        md.confirm_before_record = false;
    }
    if let Some(s) = min_silence_seconds {
        md.min_silence_seconds = s;
    }
    if device.is_some() {
        md.input_device = device;
    }

    if !config.meeting_detection.enabled {
        eprintln!(
            "Meeting detection is disabled in config. Enable it with `meeting_detection.enabled = true` in {}.",
            ScribaConfig::config_path()?.display()
        );
        return Ok(());
    }

    let md = config.meeting_detection.clone();
    let exclude_self = watcher_excludes_self();

    println!("👀 Scriba meeting watcher is running...");
    println!("   Detection watches whether another process is using a microphone.");
    if md.auto_record && md.confirm_before_record {
        println!(
            "   A Record/Ignore dialog is shown on detection (ignored after {}s if unanswered).",
            md.confirm_timeout_seconds
        );
    }
    if md.auto_record && !exclude_self {
        println!(
            "   Note: this system can't attribute mic use per process, so recordings stop after {}s of silence instead of on mic release.",
            md.min_silence_seconds
        );
    }
    println!("   Press Ctrl+C to stop.");
    if verbose {
        println!(
            "   auto_record={} confirm={} exclude_self={} silence_fallback={}s cooldown={}s ignored={:?}",
            md.auto_record,
            md.confirm_before_record,
            exclude_self,
            md.min_silence_seconds,
            md.cooldown_seconds,
            md.ignored_processes
        );
    }

    // First Ctrl+C requests a graceful shutdown; a second force-quits.
    let (shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(false);
    tokio::spawn(async move {
        let _ = tokio::signal::ctrl_c().await;
        println!("\n👋 Stopping meeting watcher (Ctrl+C again to force quit)...");
        let _ = shutdown_tx.send(true);
        let _ = tokio::signal::ctrl_c().await;
        eprintln!("Aborting.");
        std::process::exit(130);
    });

    run_autopilot(
        config,
        AutopilotOptions {
            transcribe: !no_transcribe,
            once,
            verbose,
            quiet: false,
        },
        shutdown_rx,
        Arc::new(RecordingStatus::default()),
    )
    .await
}
