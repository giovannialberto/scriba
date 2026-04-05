//! High-level workflow orchestration for Scriba.

use anyhow::{Context, Result};
use std::path::{Path, PathBuf};
use std::time::Duration;
use tokio::sync::mpsc;

use super::audio::CompressionSettings;
use super::config::{ScribaConfig, TranscriptionMode};
use super::files::FileManager;
use super::recording::{record_audio, RecordOptions};
use super::transcription::transcribe_audio;
use super::types::{ManagedRecording, RecordingConfig, RecordingMode};
use crate::database::{Database, Recording};
use crate::enrichment::{EnrichmentService, WorldContext, WorldData, WorldEntityExtractionResult};
use crate::enrichment::world::{OrgInfo, PersonInfo, ProjectInfo};
use crate::entities::{EntityLinker, EntityRegistry};
use crate::utils::{generate_recording_name, BASE_PATH};
use chrono::Utc;

/// Unified database operations with consistent error handling.
pub struct DatabaseManager {
    db: Database,
}

impl DatabaseManager {
    pub fn new() -> Result<Self> {
        Ok(Self {
            db: Database::new().context("Failed to connect to database")?,
        })
    }

    /// Create and insert a recording from ManagedRecording.
    pub fn save_recording(&mut self, recording: &ManagedRecording) -> Result<i64> {
        let db_recording = Recording {
            id: None,
            directory_name: recording.directory_name.clone(),
            display_name: recording.display_name.clone(),
            created_at: Utc::now(),
            updated_at: Utc::now(),
            duration_seconds: recording.metadata.duration_seconds,
            file_size_bytes: recording.metadata.file_size_bytes,
            audio_format: recording.metadata.audio_format.clone(),
            sample_rate: recording.metadata.sample_rate,
            channels: recording.metadata.channels,
            has_transcript: recording.transcript_path.is_some(),
            transcript_status: if recording.transcript_path.is_some() {
                "completed".to_string()
            } else {
                "pending".to_string()
            },
            language_code: "auto".to_string(),
            model_used: "whisper.cpp".to_string(),
            tags: None,
            summary: None,
            key_points: None,
            action_items: None,
            speakers: None,
            sentiment_score: None,
            search_index: None,
            categories: None,
            confidence_score: None,
            audio_path: recording
                .audio_path
                .file_name()
                .unwrap()
                .to_string_lossy()
                .to_string(),
            transcript_path: recording
                .transcript_path
                .as_ref()
                .and_then(|p| p.file_name())
                .map(|name| name.to_string_lossy().to_string()),
        };

        self.db
            .insert_recording(&db_recording)
            .context("Failed to insert recording into database")
    }

    /// Update recording with transcription info.
    pub fn update_transcription(
        &mut self,
        directory_name: &str,
        _transcript_path: &Path,
        model_used: &str,
    ) -> Result<()> {
        if let Some(rec) = self
            .db
            .get_recording_by_directory(directory_name)
            .context("Failed to query recording by directory name")?
        {
            if let Some(id) = rec.id {
                self.db
                    .update_recording_transcript_status_and_model(id, "completed", true, model_used)
                    .context("Failed to update transcript status/model in database")?;
            }
        }
        Ok(())
    }
}

/// High-level workflow orchestrator with configuration management.
pub struct WorkflowManager {
    db_manager: DatabaseManager,
    config: ScribaConfig,
}

impl WorkflowManager {
    /// Create new workflow manager with loaded configuration.
    pub fn new() -> Result<Self> {
        Ok(Self {
            db_manager: DatabaseManager::new()?,
            config: ScribaConfig::load().context("Failed to load Scriba configuration")?,
        })
    }

    /// Create workflow manager with custom configuration.
    pub fn with_config(config: ScribaConfig) -> Result<Self> {
        Ok(Self {
            db_manager: DatabaseManager::new()?,
            config,
        })
    }

    /// Get reference to the configuration.
    pub fn config(&self) -> &ScribaConfig {
        &self.config
    }

    /// Complete recording workflow: record -> save -> optionally transcribe.
    pub async fn complete_recording_workflow(
        &mut self,
        config: RecordingConfig,
        mode: RecordingMode,
    ) -> Result<ManagedRecording> {
        let directory_name = generate_recording_name(config.name.clone());
        let recording_path = PathBuf::from(&directory_name);

        let input_device = self.config.audio_settings.input_device.clone();
        let loopback_device = self.config.audio_settings.loopback_device.clone();
        let final_directory_name = match mode {
            RecordingMode::Cli => {
                let result = record_audio(
                    recording_path,
                    RecordOptions {
                        compression_settings: config.compression.clone(),
                        stop_rx: None,
                        level_tx: None,
                        verbose: true,
                        silence_timeout: None,
                        input_device,
                        loopback_device,
                    },
                )
                .await?;
                result.recording_name
            }
            RecordingMode::Tui { stop_rx, level_tx, silence_timeout } => {
                let result = record_audio(
                    recording_path,
                    RecordOptions {
                        compression_settings: config.compression.clone(),
                        stop_rx: Some(stop_rx),
                        level_tx: Some(level_tx),
                        verbose: false,
                        silence_timeout,
                        input_device,
                        loopback_device,
                    },
                )
                .await?;
                result.recording_name
            }
        };

        let recording_dir = BASE_PATH.join(&final_directory_name);
        let audio_path = FileManager::find_audio_file(&recording_dir)
            .ok_or_else(|| anyhow::anyhow!("No audio file found after recording"))?;

        let metadata = FileManager::extract_audio_metadata(&audio_path)?;

        let mut recording = ManagedRecording {
            directory_name: final_directory_name,
            display_name: config.name,
            audio_path,
            transcript_path: None,
            metadata,
        };

        if config.auto_transcribe {
            if let Some(transcription_mode) = config.transcription_mode {
                println!("📝 Starting auto-transcription...");
                recording = self
                    .transcribe_recording(recording, transcription_mode)
                    .await?;
            }
        }

        Ok(recording)
    }

    /// Complete import workflow: copy -> save -> optionally transcribe.
    pub async fn complete_import_workflow(
        &mut self,
        source_file: &Path,
        display_name: Option<String>,
        transcription_mode: Option<TranscriptionMode>,
    ) -> Result<ManagedRecording> {
        self.complete_import_workflow_internal(source_file, display_name, transcription_mode, true)
            .await
    }

    /// Silent version of complete_import_workflow for TUI usage.
    pub async fn complete_import_workflow_silent(
        &mut self,
        source_file: &Path,
        display_name: Option<String>,
        transcription_mode: Option<TranscriptionMode>,
    ) -> Result<ManagedRecording> {
        self.complete_import_workflow_internal(source_file, display_name, transcription_mode, false)
            .await
    }

    async fn complete_import_workflow_internal(
        &mut self,
        source_file: &Path,
        display_name: Option<String>,
        transcription_mode: Option<TranscriptionMode>,
        verbose: bool,
    ) -> Result<ManagedRecording> {
        let recording_dir = FileManager::create_recording_directory(display_name.clone())?;
        let mut recording =
            FileManager::import_audio_file(source_file, &recording_dir, display_name)?;

        let recording_id = self.db_manager.save_recording(&recording)?;
        if verbose {
            println!("✅ File imported to database with ID: {}", recording_id);
        }

        if let Some(mode) = transcription_mode {
            if verbose {
                println!("📝 Starting transcription...");
            }
            recording = self
                .transcribe_recording_internal(recording, mode, verbose)
                .await?;
        }

        FileManager::cleanup_temp_files(&recording_dir)?;
        Ok(recording)
    }

    async fn transcribe_recording_internal(
        &mut self,
        mut recording: ManagedRecording,
        mode: TranscriptionMode,
        verbose: bool,
    ) -> Result<ManagedRecording> {
        let directory_path = PathBuf::from(&recording.directory_name);
        transcribe_audio(&directory_path, Some(mode), verbose)
            .await
            .context("Transcription failed")?;

        let transcript_path = BASE_PATH
            .join(&recording.directory_name)
            .join("transcript.txt");
        if transcript_path.exists() {
            recording.transcript_path = Some(transcript_path.clone());

            // Run enrichment if enabled
            if self.config.enrichment.enabled {
                if let Err(e) = self
                    .enrich_recording(&recording.directory_name, &transcript_path, verbose)
                    .await
                {
                    if verbose {
                        eprintln!("⚠️ Enrichment failed (non-fatal): {}", e);
                    }
                }
            }
        }
        Ok(recording)
    }

    /// Enrich a recording with AI-extracted metadata.
    pub async fn enrich_recording(
        &mut self,
        directory_name: &str,
        transcript_path: &Path,
        verbose: bool,
    ) -> Result<()> {
        let config = &self.config.enrichment;

        if verbose {
            println!("🧠 Starting knowledge extraction...");
        }

        // Read transcript content
        let transcript_content = std::fs::read_to_string(transcript_path)
            .context("Failed to read transcript for enrichment")?;

        if transcript_content.trim().is_empty() {
            if verbose {
                println!("⚠️ Transcript is empty, skipping enrichment");
            }
            return Ok(());
        }

        // Create enrichment service from config
        let service = EnrichmentService::from_config(config);

        // Check if the enrichment provider is available
        if let Err(e) = service.health_check().await {
            return Err(anyhow::anyhow!(
                "Enrichment provider ({}) not available: {}",
                service.provider_display_name(),
                e,
            ));
        }

        if verbose {
            println!("  📊 Extracting metadata from transcript...");
        }

        // Rebuild world from entities BEFORE extraction so LLM sees
        // up-to-date aliases and merged entity state
        if let Err(e) = rebuild_world_from_entities(&self.db_manager.db) {
            if verbose {
                eprintln!("  ⚠️ Failed to refresh world context: {}", e);
            }
        }

        // Load world context (Scriba's understanding of the owner)
        let world = WorldContext::load().unwrap_or_default();
        let world_content = if world.has_content() {
            if verbose {
                println!("  🌍 Using world context for extraction");
            }
            Some(world.content.clone())
        } else {
            None
        };

        // Extract metadata with world context — the LLM resolves entities inline
        let mut extraction = service
            .extract_with_full_context(
                &transcript_content,
                world_content.as_deref(),
            )
            .await
            .context("Failed to extract metadata from transcript")?;

        // Web search verification for unresolved entities
        if config.search_enabled {
            let has_unresolved = extraction.people.iter().any(|e| e.resolved_to.is_none())
                || extraction.organizations.iter().any(|e| e.resolved_to.is_none());

            if has_unresolved {
                if verbose {
                    println!("  🔍 Searching web for unresolved entities...");
                }

                let search_results = crate::enrichment::search::search_unresolved_entities(
                    &extraction,
                    config.max_search_results,
                )
                .await;

                if !search_results.is_empty() {
                    // Build world summary for the resolution prompt
                    let world_summary = world
                        .parsed()
                        .map(|d| d.entities_summary())
                        .unwrap_or_default();

                    let resolutions = service
                        .resolve_with_search(&search_results, &world_summary)
                        .await;

                    if !resolutions.is_empty() {
                        let resolved_count = apply_search_resolutions(&mut extraction, &resolutions);
                        if verbose && resolved_count > 0 {
                            println!("  ✅ Search resolved {} entities", resolved_count);
                        }
                    }
                }
            }
        }

        // Replace transcript spellings with canonical names everywhere
        let mut title = extraction.title.clone();
        let mut summary = extraction.summary.clone();
        let mut corrected_transcript = transcript_content.clone();
        for (entity, _) in extraction.all_entities() {
            if let Some(canonical) = &entity.resolved_to {
                if entity.name != *canonical {
                    title = title.replace(&entity.name, canonical);
                    summary = summary.replace(&entity.name, canonical);
                    corrected_transcript = corrected_transcript.replace(&entity.name, canonical);
                }
            }
        }

        if verbose {
            println!("  ✅ Extracted: title='{}', {} topics, {} people, {} organizations",
                title,
                extraction.topics.len(),
                extraction.people.len(),
                extraction.organizations.len()
            );
        }

        // Get recording from database
        let recording = self.db_manager.db.get_recording_by_directory(directory_name)?
            .ok_or_else(|| anyhow::anyhow!("Recording not found: {}", directory_name))?;
        let recording_id = recording.id
            .ok_or_else(|| anyhow::anyhow!("Recording has no ID"))?;

        // Update recording with enrichment data
        let key_points_json = if extraction.key_points.is_empty() {
            None
        } else {
            Some(serde_json::to_string(&extraction.key_points)?)
        };

        let action_items_json = if extraction.action_items.is_empty() {
            None
        } else {
            Some(serde_json::to_string(&extraction.action_items)?)
        };

        self.db_manager.db.update_recording_enrichment(
            recording_id,
            Some(title.as_str()),
            Some(summary.as_str()),
            key_points_json.as_deref(),
            action_items_json.as_deref(),
        )?;

        // Update transcript with topics and entities
        let topics_json = if extraction.topics.is_empty() {
            None
        } else {
            Some(serde_json::to_string(&extraction.topics)?)
        };

        // Combine people and organizations into entities JSON
        let entities: Vec<serde_json::Value> = extraction
            .people
            .iter()
            .map(|p| {
                let name = p.resolved_to.as_deref().unwrap_or(&p.name);
                serde_json::json!({"type": "person", "name": name, "context": p.context})
            })
            .chain(
                extraction
                    .organizations
                    .iter()
                    .map(|o| {
                        let name = o.resolved_to.as_deref().unwrap_or(&o.name);
                        serde_json::json!({"type": "organization", "name": name, "context": o.context})
                    }),
            )
            .collect();

        let entities_json = if entities.is_empty() {
            None
        } else {
            Some(serde_json::to_string(&entities)?)
        };

        self.db_manager.db.update_transcript_enrichment(
            recording_id,
            entities_json.as_deref(),
            topics_json.as_deref(),
        )?;

        // Update transcript with corrected names (both DB and file on disk)
        if corrected_transcript != transcript_content {
            self.db_manager.db.upsert_transcript(recording_id, &corrected_transcript)?;
            let _ = std::fs::write(transcript_path, &corrected_transcript);
            if verbose {
                println!("  ✏️ Transcript updated with canonical entity names");
            }
        }

        // Process entity linking — entities are already resolved by the LLM
        if verbose {
            println!("  🔗 Linking entities...");
        }

        let linker = EntityLinker::new();
        let report = linker
            .process_extraction(&mut self.db_manager.db, recording_id, &extraction)?;

        if verbose {
            println!(
                "  ✅ Entity linking complete: {} linked, {} created",
                report.linked_existing, report.created_new
            );
        }

        // Speaker identification: resolve generic labels to real names
        // Speaker diarization removed — may be reintroduced in a future release.

        // Evolve world: LLM produces a delta, we apply it to entities first,
        // then rebuild world.md from entities (entities are the source of truth).
        if config.evolve_world && world.has_content() {
            if verbose {
                println!("  🌍 Evolving world understanding...");
            }

            // Get current world JSON for the evolution prompt
            let world_json = if let Some(data) = world.parsed() {
                data.to_json().unwrap_or_default()
            } else {
                // Migrate legacy world to structured format
                if verbose {
                    println!("  🔄 Migrating world to structured format...");
                }
                match service.extract_world_seed(&world.content).await {
                    Ok(data) => data.to_json().unwrap_or_default(),
                    Err(_) => world.content.clone(),
                }
            };

            match service
                .evolve_world(&world_json, &transcript_content, &extraction)
                .await
            {
                Ok(Some(delta)) => {
                    // Apply delta to entities (source of truth) with LLM context compaction
                    if let Err(e) = apply_world_delta_to_entities(&mut self.db_manager.db, &delta, &service, verbose).await {
                        if verbose {
                            eprintln!("  ⚠️ Failed to apply delta to entities: {}", e);
                        }
                    }
                    // Rebuild world.md from entities
                    if let Err(e) = rebuild_world_from_entities(&self.db_manager.db) {
                        if verbose {
                            eprintln!("  ⚠️ Failed to rebuild world: {}", e);
                        }
                    } else if verbose {
                        println!("  ✅ World description updated");
                    }
                }
                Ok(None) => {
                    if verbose {
                        println!("  ⚠️ No new changes from this recording");
                    }
                }
                Err(e) => {
                    if verbose {
                        eprintln!("  ⚠️ Failed to evolve world (non-fatal): {}", e);
                    }
                }
            }
        }

        if verbose {
            println!("✅ Knowledge extraction complete!");
        }

        Ok(())
    }

    /// Enrich an existing recording (for manual enrichment or re-enrichment).
    pub async fn enrich_existing_recording(
        &mut self,
        directory_name: &str,
        verbose: bool,
    ) -> Result<()> {
        let recording_dir = BASE_PATH.join(directory_name);
        let transcript_path = recording_dir.join("transcript.txt");

        if !transcript_path.exists() {
            return Err(anyhow::anyhow!(
                "No transcript found for recording: {}",
                directory_name
            ));
        }

        self.enrich_recording(directory_name, &transcript_path, verbose)
            .await
    }

    /// Unified transcription workflow that works with any ManagedRecording (verbose).
    pub async fn transcribe_recording(
        &mut self,
        recording: ManagedRecording,
        mode: TranscriptionMode,
    ) -> Result<ManagedRecording> {
        self.transcribe_recording_internal(recording, mode, true)
            .await
    }

    /// Silent version of transcribe_recording for TUI usage.
    pub async fn transcribe_recording_silent(
        &mut self,
        recording: ManagedRecording,
        mode: TranscriptionMode,
    ) -> Result<ManagedRecording> {
        self.transcribe_recording_internal(recording, mode, false)
            .await
    }

    /// Convenience method for simple CLI recording.
    pub async fn record_cli(
        &mut self,
        name: Option<String>,
        compression: Option<CompressionSettings>,
        auto_transcribe: bool,
        transcription_mode: Option<TranscriptionMode>,
    ) -> Result<ManagedRecording> {
        let config = RecordingConfig {
            name,
            compression,
            auto_transcribe,
            transcription_mode,
        };

        self.complete_recording_workflow(config, RecordingMode::Cli)
            .await
    }

    /// Convenience method for TUI recording with control channels.
    pub async fn record_tui(
        &mut self,
        name: Option<String>,
        compression: Option<CompressionSettings>,
        stop_rx: mpsc::Receiver<()>,
        level_tx: mpsc::Sender<f32>,
        auto_transcribe: bool,
        transcription_mode: Option<TranscriptionMode>,
        silence_timeout: Option<Duration>,
    ) -> Result<ManagedRecording> {
        let config = RecordingConfig {
            name,
            compression,
            auto_transcribe,
            transcription_mode,
        };

        let mode = RecordingMode::Tui { stop_rx, level_tx, silence_timeout };
        self.complete_recording_workflow(config, mode).await
    }

    /// Re-transcribe an existing recording with a different model.
    pub async fn retranscribe_recording(
        &mut self,
        directory_name: &str,
        transcription_mode: TranscriptionMode,
    ) -> Result<()> {
        self.retranscribe_internal(directory_name, transcription_mode, true)
            .await
    }

    /// Silent version of retranscribe_recording for TUI usage.
    pub async fn retranscribe_recording_silent(
        &mut self,
        directory_name: &str,
        transcription_mode: TranscriptionMode,
    ) -> Result<()> {
        self.retranscribe_internal(directory_name, transcription_mode, false)
            .await
    }

    async fn retranscribe_internal(
        &mut self,
        directory_name: &str,
        transcription_mode: TranscriptionMode,
        verbose: bool,
    ) -> Result<()> {
        let recording_dir = BASE_PATH.join(directory_name);
        if !recording_dir.exists() {
            return Err(anyhow::anyhow!(
                "Recording directory not found: {}",
                directory_name
            ));
        }

        let audio_path = FileManager::find_audio_file(&recording_dir)
            .ok_or_else(|| anyhow::anyhow!("No audio file found in recording directory"))?;

        let metadata = FileManager::extract_audio_metadata(&audio_path)?;
        let recording = ManagedRecording {
            directory_name: directory_name.to_string(),
            display_name: None,
            audio_path,
            transcript_path: None,
            metadata,
        };

        let _ = self
            .transcribe_recording_internal(recording, transcription_mode, verbose)
            .await?;
        if verbose {
            println!("✅ Re-transcription complete for: {}", directory_name);
        }
        Ok(())
    }

    /// Batch import multiple audio files with progress reporting.
    pub async fn batch_import_workflow(
        &mut self,
        files: &[PathBuf],
        transcription_mode: Option<TranscriptionMode>,
        progress_callback: Option<Box<dyn Fn(usize, usize, &str) + Send>>,
    ) -> Result<Vec<ManagedRecording>> {
        let mut recordings = Vec::new();
        let total_files = files.len();

        for (index, file_path) in files.iter().enumerate() {
            if let Some(ref callback) = progress_callback {
                callback(
                    index + 1,
                    total_files,
                    &format!("Processing {}", file_path.display()),
                );
            }

            println!(
                "📁 [{}/{}] Importing: {}",
                index + 1,
                total_files,
                file_path.display()
            );

            match self
                .complete_import_workflow(file_path, None, transcription_mode.clone())
                .await
            {
                Ok(recording) => {
                    recordings.push(recording);
                    println!("✅ Successfully imported: {}", file_path.display());
                }
                Err(e) => {
                    eprintln!("❌ Failed to import {}: {}", file_path.display(), e);
                }
            }
        }

        println!(
            "🎉 Batch import complete! {} of {} files processed successfully.",
            recordings.len(),
            total_files
        );
        Ok(recordings)
    }

    /// Health check for the workflow manager and its dependencies.
    pub fn health_check(&self) -> Result<HealthStatus> {
        let mut issues = Vec::new();
        let mut warnings = Vec::new();

        if let Err(e) = DatabaseManager::new() {
            issues.push(format!("Database connection failed: {}", e));
        }

        if !BASE_PATH.exists() {
            issues.push("Base recordings directory does not exist".to_string());
        } else if let Err(e) = std::fs::create_dir_all(&*BASE_PATH) {
            issues.push(format!("Cannot write to recordings directory: {}", e));
        }

        if std::process::Command::new("ffprobe")
            .arg("-version")
            .output()
            .is_err()
        {
            warnings.push(
                "ffprobe not available - audio metadata extraction will be limited".to_string(),
            );
        }

        let models_dir = BASE_PATH.join("models");
        if !models_dir.exists() {
            warnings.push(
                "No whisper models directory found - local transcription may fail".to_string(),
            );
        }

        // Check enrichment provider availability
        if self.config.enrichment.enabled {
            // We can't await here since health_check is not async, so we just note the config
            warnings.push(format!(
                "Enrichment enabled - provider: {}, model: '{}'",
                self.config.enrichment.provider_display_name(),
                self.config.enrichment.model_name()
            ));
        }

        // Enumerate audio input devices
        let input_devices = super::recording::list_input_devices().unwrap_or_default();
        if input_devices.is_empty() {
            issues.push("No audio input devices found".to_string());
        }

        let configured_device = self.config.audio_settings.input_device.clone();

        // Detect loopback sources
        let loopback_sources = super::loopback::detect_loopback_sources().unwrap_or_default();
        let configured_loopback = self.config.audio_settings.loopback_device.clone();

        if let Some(ref lb_device) = configured_loopback {
            if !lb_device.is_empty() && loopback_sources.is_empty() {
                warnings.push(format!(
                    "Loopback device '{}' configured but no loopback sources detected",
                    lb_device
                ));
            }
        }

        let status = if issues.is_empty() {
            HealthStatusLevel::Healthy
        } else {
            HealthStatusLevel::Unhealthy
        };

        Ok(HealthStatus {
            status,
            issues,
            warnings,
            input_devices,
            configured_device,
            loopback_sources,
            configured_loopback,
        })
    }
}

/// Rebuild world.md entirely from the entity database.
///
/// Entities are the source of truth. The world file is a derived view
/// used to give the LLM context during extraction.
pub fn rebuild_world_from_entities(db: &Database) -> Result<()> {
    let all_entities = db.list_entities(None, None)?;

    let mut world = WorldData::default();

    for entity in &all_entities {
        let aliases = entity.aliases_list();
        let context = entity.context.clone().unwrap_or_default();

        match entity.entity_type.as_str() {
            "person" => {
                // Check if this is the owner
                let is_owner = context.contains("Owner of this Scriba instance");

                if is_owner {
                    world.owner.name = entity.canonical_name.clone();
                    world.owner.aliases = aliases;
                    // Parse role and organization from context
                    // Context format: "Owner of this Scriba instance. CTO at Exein"
                    let parts: Vec<&str> = context.splitn(2, '.').collect();
                    if parts.len() > 1 {
                        let role_part = parts[1].trim();
                        if let Some(at_idx) = role_part.find(" at ") {
                            world.owner.role = role_part[..at_idx].to_string();
                            world.owner.organization = role_part[at_idx + 4..].to_string();
                        } else {
                            world.owner.role = role_part.to_string();
                        }
                    }
                } else {
                    world.people.push(PersonInfo {
                        name: entity.canonical_name.clone(),
                        relationship: context,
                        aliases,
                    });
                }
            }
            "organization" => {
                world.organizations.push(OrgInfo {
                    name: entity.canonical_name.clone(),
                    description: context,
                    aliases,
                });
            }
            "project" => {
                world.projects.push(ProjectInfo {
                    name: entity.canonical_name.clone(),
                    description: context,
                });
            }
            _ => {}
        }
    }

    // Preserve interests and beliefs from existing world (no entity backing)
    if let Ok(current_world) = WorldContext::load() {
        if let Some(current_data) = current_world.parsed() {
            world.interests = current_data.interests;
            world.beliefs = current_data.beliefs;
            // Preserve owner location if not derivable from entities
            if world.owner.location.is_empty() {
                world.owner.location = current_data.owner.location;
            }
        }
    }

    let mut world_ctx = WorldContext::load().unwrap_or_default();
    world_ctx.update_data(&world)?;

    Ok(())
}

/// Apply a world evolution delta directly to the entity database.
///
/// Instead of merging into world.md first, we apply changes to entities
/// (the source of truth), then rebuild the world from them.
///
/// Uses LLM-powered context compaction: existing context + new info are merged
/// into a clean, self-contained description in a single LLM call.
/// Falls back to simple append if the compaction call fails.
async fn apply_world_delta_to_entities(
    db: &mut Database,
    delta: &WorldData,
    service: &EnrichmentService,
    verbose: bool,
) -> Result<()> {
    let mut registry = EntityRegistry::new(db);

    // Apply owner changes
    if !delta.owner.name.is_empty() {
        if let Some(entity) = registry.get_entity_by_name_or_alias(&delta.owner.name)? {
            let entity_id = entity.id.unwrap();
            for alias in &delta.owner.aliases {
                registry.add_entity_alias(entity_id, alias)?;
            }
        }
    }

    // Generic placeholder names that should never become entities
    let blocked_names: &[&str] = &[
        "speaker", "speakers", "narrator", "author", "autore",
        "owner", "user", "host", "interviewer", "interviewee",
        "participant", "moderator", "presenter", "listener",
        "the speaker", "the owner", "the author", "the narrator",
        "you", "me", "i", "we", "they",
    ];

    // Apply people changes
    for person in &delta.people {
        if person.name.is_empty() || blocked_names.contains(&person.name.to_lowercase().as_str()) {
            continue;
        }
        match registry.get_entity_by_name_or_alias(&person.name)? {
            Some(entity) => {
                let entity_id = entity.id.unwrap();
                if !person.relationship.is_empty() {
                    let existing_ctx = entity.context.unwrap_or_default();
                    let compacted = compact_or_append(
                        service,
                        &person.name,
                        "person",
                        &existing_ctx,
                        &person.relationship,
                        verbose,
                    ).await;
                    registry.update_entity_context(entity_id, &compacted)?;
                }
                for alias in &person.aliases {
                    registry.add_entity_alias(entity_id, alias)?;
                }
            }
            None => {
                let ctx = if person.relationship.is_empty() {
                    None
                } else {
                    Some(person.relationship.as_str())
                };
                let new_entity = registry.create_entity("person", &person.name, ctx)?;
                if let Some(entity_id) = new_entity.id {
                    for alias in &person.aliases {
                        registry.add_entity_alias(entity_id, alias)?;
                    }
                }
            }
        }
    }

    // Apply organization changes
    for org in &delta.organizations {
        if org.name.is_empty() {
            continue;
        }
        match registry.get_entity_by_name_or_alias(&org.name)? {
            Some(entity) => {
                let entity_id = entity.id.unwrap();
                if !org.description.is_empty() {
                    let existing_ctx = entity.context.unwrap_or_default();
                    let compacted = compact_or_append(
                        service,
                        &org.name,
                        "organization",
                        &existing_ctx,
                        &org.description,
                        verbose,
                    ).await;
                    registry.update_entity_context(entity_id, &compacted)?;
                }
                for alias in &org.aliases {
                    registry.add_entity_alias(entity_id, alias)?;
                }
            }
            None => {
                let ctx = if org.description.is_empty() {
                    None
                } else {
                    Some(org.description.as_str())
                };
                let new_entity = registry.create_entity("organization", &org.name, ctx)?;
                if let Some(entity_id) = new_entity.id {
                    for alias in &org.aliases {
                        registry.add_entity_alias(entity_id, alias)?;
                    }
                }
            }
        }
    }

    Ok(())
}

/// Compact entity context via LLM, falling back to simple append on failure.
async fn compact_or_append(
    service: &EnrichmentService,
    entity_name: &str,
    entity_type: &str,
    existing_context: &str,
    new_info: &str,
    verbose: bool,
) -> String {
    if existing_context.is_empty() {
        return new_info.to_string();
    }

    // Skip compaction if the new info adds nothing beyond what's already known
    if existing_context.trim() == new_info.trim()
        || existing_context.to_lowercase().contains(&new_info.to_lowercase())
    {
        return existing_context.to_string();
    }

    // If the new info is a superset of existing, just use it directly
    if new_info.to_lowercase().contains(&existing_context.to_lowercase()) {
        return new_info.to_string();
    }

    match service
        .compact_entity_context(entity_name, entity_type, existing_context, new_info)
        .await
    {
        Ok(compacted) => {
            if verbose {
                println!("    📝 Compacted context for '{}'", entity_name);
            }
            compacted
        }
        Err(e) => {
            if verbose {
                eprintln!("    ⚠️ Compaction failed for '{}', appending: {}", entity_name, e);
            }
            // Fallback: simple append
            format!("{}. {}", existing_context.trim_end_matches('.'), new_info)
        }
    }
}

/// Apply search resolutions to an extraction result.
///
/// For each entity in the extraction whose name matches a key in `resolutions`,
/// sets its `resolved_to` to the resolved value. Returns the count of entities updated.
fn apply_search_resolutions(
    extraction: &mut crate::enrichment::ExtractionResult,
    resolutions: &std::collections::HashMap<String, Option<String>>,
) -> usize {
    let mut count = 0;

    for entity in &mut extraction.people {
        if let Some(resolved) = resolutions.get(&entity.name) {
            if resolved.is_some() {
                entity.resolved_to = resolved.clone();
                count += 1;
            }
        }
    }

    for entity in &mut extraction.organizations {
        if let Some(resolved) = resolutions.get(&entity.name) {
            if resolved.is_some() {
                entity.resolved_to = resolved.clone();
                count += 1;
            }
        }
    }

    count
}

/// Initialize a world from seed content (used by both CLI and TUI onboarding).
///
/// Does: health check → extract_world_seed → save world.md → extract_world_entities → create entities.
/// Returns `Ok(Some((world_data, entities)))` on success, `Ok(None)` if Ollama unavailable (raw seed saved).
pub async fn initialize_world_from_seed(
    db: &mut Database,
    config: &ScribaConfig,
    seed_content: &str,
) -> Result<Option<(WorldData, WorldEntityExtractionResult)>> {
    let service = EnrichmentService::from_config(&config.enrichment);

    // Check provider availability
    if service.health_check().await.is_err() {
        // Save raw seed as world.md and return None
        WorldContext::initialize(seed_content)?;
        return Ok(None);
    }

    // Convert seed to structured JSON via LLM
    let world_data = match service.extract_world_seed(seed_content).await {
        Ok(data) => data,
        Err(_) => {
            // Fallback: save raw seed
            WorldContext::initialize(seed_content)?;
            return Ok(None);
        }
    };

    let world_json = world_data.to_json().unwrap_or_else(|_| seed_content.to_string());
    WorldContext::initialize(&world_json)?;

    // Extract entities from the seed
    let extraction = match service.extract_world_entities(seed_content).await {
        Ok(ext) => ext,
        Err(_) => {
            return Ok(Some((world_data, WorldEntityExtractionResult {
                people: vec![],
                organizations: vec![],
            })));
        }
    };

    // Create entities in DB
    let mut registry = EntityRegistry::new(db);

    for person in &extraction.people {
        let context = if person.is_owner {
            format!("Owner of this Scriba instance. {}", person.context)
        } else {
            person.context.clone()
        };

        if let Ok(entity) = registry.create_entity("person", &person.name, Some(&context)) {
            if let Some(entity_id) = entity.id {
                for alias in &person.aliases {
                    let _ = registry.add_entity_alias(entity_id, alias);
                }
            }
        }
    }

    for org in &extraction.organizations {
        if let Ok(entity) = registry.create_entity("organization", &org.name, Some(&org.context)) {
            if let Some(entity_id) = entity.id {
                for alias in &org.aliases {
                    let _ = registry.add_entity_alias(entity_id, alias);
                }
            }
        }
    }

    Ok(Some((world_data, extraction)))
}

/// Health status for workflow manager.
#[derive(Debug)]
pub struct HealthStatus {
    pub status: HealthStatusLevel,
    pub issues: Vec<String>,
    pub warnings: Vec<String>,
    pub input_devices: Vec<String>,
    pub configured_device: Option<String>,
    pub loopback_sources: Vec<String>,
    pub configured_loopback: Option<String>,
}

#[derive(Debug)]
pub enum HealthStatusLevel {
    Healthy,
    Unhealthy,
}

impl HealthStatus {
    pub fn is_healthy(&self) -> bool {
        matches!(self.status, HealthStatusLevel::Healthy)
    }

    pub fn print_report(&self) {
        match self.status {
            HealthStatusLevel::Healthy => {
                println!("✅ Scriba health check: All systems operational");
            }
            HealthStatusLevel::Unhealthy => {
                println!("❌ Scriba health check: Issues detected");
            }
        }

        if !self.issues.is_empty() {
            println!("\n🚨 Issues:");
            for issue in &self.issues {
                println!("  - {}", issue);
            }
        }

        if !self.warnings.is_empty() {
            println!("\n⚠️ Warnings:");
            for warning in &self.warnings {
                println!("  - {}", warning);
            }
        }

        if !self.input_devices.is_empty() {
            println!("\n🎙️ Audio input devices:");
            for device in &self.input_devices {
                let marker = match &self.configured_device {
                    Some(configured) if device.to_lowercase().contains(&configured.to_lowercase()) => " ← configured",
                    _ => "",
                };
                println!("  - {}{}", device, marker);
            }
            if let Some(ref configured) = self.configured_device {
                println!("\n  Configured device: \"{}\"", configured);
            } else {
                println!("\n  No device configured (using system default).");
                println!("  To use a specific device (e.g. headphones), set it in config:");
                println!("    Edit ~/scriba_recordings/config.json and set");
                println!("    \"input_device\": \"<device name>\" in audio_settings,");
                println!("    or pass --device \"<name>\" to `scriba record`.");
            }
        }

        // Loopback sources
        println!("\n🔊 System audio loopback:");
        if self.loopback_sources.is_empty() {
            #[cfg(target_os = "macos")]
            println!("  ScreenCaptureKit not available (requires macOS 12.3+)");
            #[cfg(target_os = "linux")]
            println!("  No PulseAudio/PipeWire monitor sources found");
            #[cfg(not(any(target_os = "macos", target_os = "linux")))]
            println!("  Loopback not supported on this platform");
        } else {
            for source in &self.loopback_sources {
                println!("  - {}", source);
            }
        }
        match &self.configured_loopback {
            Some(d) if d.is_empty() => {
                println!("  Loopback: enabled (auto-detect)");
            }
            Some(d) => {
                println!("  Loopback: enabled (device: \"{}\")", d);
            }
            None => {
                println!("  Loopback: disabled");
                println!("  Enable with --loopback flag or set \"loopback_device\": \"\" in config.");
            }
        }
    }
}
