//! File operations for Scriba recordings.

use anyhow::{Context, Result};
use std::path::{Path, PathBuf};

use super::types::{ManagedRecording, RecordingMetadata};
use crate::utils::BASE_PATH;

/// Supported audio file extensions.
pub const AUDIO_EXTENSIONS: [&str; 10] = [
    "wav", "mp3", "m4a", "aac", "flac", "ogg", "opus", "aiff", "aif", "caf",
];

/// Common recording file names to look for.
pub const RECORDING_FILES: [&str; 6] = [
    "recording.wav",
    "recording.mp3",
    "recording.m4a",
    "recording.flac",
    "recording.ogg",
    "recording.aac",
];

/// Unified file operations for Scriba recordings.
pub struct FileManager;

impl FileManager {
    /// Create a new recording directory with proper structure.
    pub fn create_recording_directory(name: Option<String>) -> Result<PathBuf> {
        let directory_name = crate::utils::generate_recording_name(name);
        let dir_path = BASE_PATH.join(&directory_name);
        std::fs::create_dir_all(&dir_path).with_context(|| {
            format!(
                "Failed to create recording directory: {}",
                dir_path.display()
            )
        })?;
        Ok(dir_path)
    }

    /// Import external audio file into Scriba structure.
    pub fn import_audio_file(
        source: &Path,
        target_directory: &Path,
        display_name: Option<String>,
    ) -> Result<ManagedRecording> {
        if !source.exists() {
            return Err(anyhow::anyhow!(
                "Source file not found: {}",
                source.display()
            ));
        }

        let file_extension = source
            .extension()
            .and_then(|ext| ext.to_str())
            .unwrap_or("wav");

        let target_file = target_directory.join(format!("recording.{}", file_extension));

        std::fs::copy(source, &target_file).with_context(|| {
            format!(
                "Failed to copy {} to {}",
                source.display(),
                target_file.display()
            )
        })?;

        let metadata = Self::extract_audio_metadata(&target_file)?;

        Ok(ManagedRecording {
            directory_name: target_directory
                .file_name()
                .unwrap()
                .to_string_lossy()
                .to_string(),
            display_name,
            audio_path: target_file,
            transcript_path: None,
            metadata,
        })
    }

    /// Extract comprehensive metadata from audio file.
    pub fn extract_audio_metadata(file_path: &Path) -> Result<RecordingMetadata> {
        if !file_path.exists() {
            return Err(anyhow::anyhow!(
                "Audio file not found: {}",
                file_path.display()
            ));
        }

        let file_metadata = std::fs::metadata(file_path)
            .with_context(|| format!("Failed to read file metadata: {}", file_path.display()))?;
        let file_size_bytes = file_metadata.len() as i64;

        let audio_format = file_path
            .extension()
            .and_then(|ext| ext.to_str())
            .unwrap_or("wav")
            .to_lowercase();

        let (sample_rate, channels, duration_seconds) = match audio_format.as_str() {
            "wav" => Self::extract_wav_metadata(file_path)?,
            "mp3" => Self::extract_mp3_metadata(file_path)?,
            "m4a" | "aac" => Self::extract_aac_metadata(file_path)?,
            _ => {
                let (sample_rate, channels) = (44100i64, 2i64);
                let duration = Self::calculate_audio_duration(file_path, sample_rate, channels)?;
                (sample_rate, channels, duration)
            }
        };

        Ok(RecordingMetadata {
            duration_seconds: Some(duration_seconds),
            file_size_bytes: Some(file_size_bytes),
            audio_format,
            sample_rate,
            channels,
        })
    }

    /// Extract metadata from WAV files using hound.
    fn extract_wav_metadata(file_path: &Path) -> Result<(i64, i64, i64)> {
        let reader = hound::WavReader::open(file_path)
            .with_context(|| format!("Failed to open WAV file: {}", file_path.display()))?;

        let spec = reader.spec();
        let sample_rate = spec.sample_rate as i64;
        let channels = spec.channels as i64;
        let duration_samples = reader.duration() as i64;
        let duration_seconds = duration_samples / sample_rate;

        Ok((sample_rate, channels, duration_seconds))
    }

    /// Extract metadata from MP3 files using ffprobe or estimation.
    fn extract_mp3_metadata(file_path: &Path) -> Result<(i64, i64, i64)> {
        if let Ok(duration) = Self::get_duration_with_ffprobe(file_path) {
            return Ok((44100, 2, duration));
        }

        let file_size = std::fs::metadata(file_path)?.len() as i64;
        let estimated_duration = (file_size * 8) / (128 * 1000);
        Ok((44100, 2, estimated_duration.max(1)))
    }

    /// Extract metadata from AAC/M4A files.
    fn extract_aac_metadata(file_path: &Path) -> Result<(i64, i64, i64)> {
        if let Ok(duration) = Self::get_duration_with_ffprobe(file_path) {
            return Ok((44100, 2, duration));
        }
        Ok((44100, 2, 1))
    }

    /// Get duration using ffprobe if available.
    fn get_duration_with_ffprobe(file_path: &Path) -> Result<i64> {
        let output = std::process::Command::new("ffprobe")
            .args([
                "-v",
                "quiet",
                "-show_entries",
                "format=duration",
                "-of",
                "csv=p=0",
            ])
            .arg(file_path)
            .output()?;

        if output.status.success() {
            let duration_str = String::from_utf8_lossy(&output.stdout);
            let duration_f64: f64 = duration_str
                .trim()
                .parse()
                .context("Failed to parse duration from ffprobe")?;
            Ok(duration_f64.round() as i64)
        } else {
            Err(anyhow::anyhow!("ffprobe failed"))
        }
    }

    /// Calculate the duration of an audio file in seconds.
    pub fn calculate_audio_duration(
        file_path: &Path,
        _sample_rate: i64,
        _channels: i64,
    ) -> Result<i64> {
        let extension = file_path
            .extension()
            .and_then(|ext| ext.to_str())
            .unwrap_or("");

        match extension.to_lowercase().as_str() {
            "wav" => {
                let reader = hound::WavReader::open(file_path)
                    .context("Failed to open WAV file for duration calculation")?;

                let spec = reader.spec();
                let wav_sample_rate = spec.sample_rate as i64;
                let duration_samples = reader.duration() as i64;
                let duration_seconds = duration_samples / wav_sample_rate;

                Ok(duration_seconds)
            }
            "mp3" => {
                let output = std::process::Command::new("ffprobe")
                    .arg("-v")
                    .arg("quiet")
                    .arg("-show_entries")
                    .arg("format=duration")
                    .arg("-of")
                    .arg("csv=p=0")
                    .arg(file_path)
                    .output();

                match output {
                    Ok(output) if output.status.success() => {
                        let duration_str = String::from_utf8_lossy(&output.stdout);
                        let duration_f64: f64 = duration_str.trim().parse().unwrap_or(1.0);
                        Ok(duration_f64.round() as i64)
                    }
                    _ => {
                        let file_size = std::fs::metadata(file_path)?.len() as i64;
                        let estimated_duration = file_size / 4000;
                        Ok(estimated_duration.max(1))
                    }
                }
            }
            _ => Ok(1),
        }
    }

    /// Clean up temporary files in recording directory.
    pub fn cleanup_temp_files(directory: &Path) -> Result<()> {
        let patterns = ["*_tmp_whisper_16k.wav", "*.tmp", "recording.wav.bak"];

        for pattern in patterns {
            if let Ok(entries) = glob::glob(&directory.join(pattern).to_string_lossy()) {
                for entry in entries.flatten() {
                    let _ = std::fs::remove_file(entry);
                }
            }
        }
        Ok(())
    }

    /// Find an audio file in a recording directory.
    /// This is the single source of truth for finding audio files.
    pub fn find_audio_file(recording_dir: &Path) -> Option<PathBuf> {
        if !recording_dir.exists() {
            return None;
        }

        // First, check for standard recording files
        for filename in RECORDING_FILES {
            let path = recording_dir.join(filename);
            if path.exists() {
                return Some(path);
            }
        }

        // Fall back to scanning directory for any audio file
        if let Ok(read_dir) = std::fs::read_dir(recording_dir) {
            for entry in read_dir.flatten() {
                let path = entry.path();
                if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
                    if AUDIO_EXTENSIONS
                        .iter()
                        .any(|x| x.eq_ignore_ascii_case(ext))
                    {
                        return Some(path);
                    }
                }
            }
        }

        None
    }

    /// Validate an audio file exists and is not corrupted.
    pub fn validate_audio_file(path: &Path) -> Result<()> {
        if !path.exists() {
            return Err(anyhow::anyhow!("Audio file not found: {}", path.display()));
        }

        let metadata = std::fs::metadata(path).context("Failed to read file metadata")?;

        if metadata.len() == 0 {
            return Err(anyhow::anyhow!("Audio file is empty"));
        }

        if metadata.len() < 1024 {
            return Err(anyhow::anyhow!(
                "Audio file too small (< 1KB), likely corrupted"
            ));
        }

        if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
            match ext.to_lowercase().as_str() {
                "wav" | "mp3" | "m4a" | "flac" | "ogg" | "aac" => Ok(()),
                _ => Err(anyhow::anyhow!(
                    "Unsupported audio format: {}. Supported: wav, mp3, m4a, flac, ogg, aac",
                    ext
                )),
            }
        } else {
            Err(anyhow::anyhow!(
                "File has no extension, cannot determine audio format"
            ))
        }
    }

    /// Resolve an audio path from various input formats.
    /// Handles absolute paths, relative paths, and directory names.
    pub fn resolve_audio_path(input_path: &PathBuf) -> Result<PathBuf> {
        if input_path.is_absolute() {
            if input_path.exists() {
                Self::validate_audio_file(input_path)?;
                return Ok(input_path.clone());
            } else {
                return Err(anyhow::anyhow!(
                    "Audio file not found: {}",
                    input_path.display()
                ));
            }
        }

        if input_path.extension().is_some() {
            let full_path = BASE_PATH.join(input_path);
            if full_path.exists() {
                Self::validate_audio_file(&full_path)?;
                return Ok(full_path);
            } else {
                return Err(anyhow::anyhow!(
                    "Audio file not found: {}",
                    full_path.display()
                ));
            }
        }

        // Handle directory name
        let recording_dir = BASE_PATH.join(input_path);

        if let Some(audio_path) = Self::find_audio_file(&recording_dir) {
            Self::validate_audio_file(&audio_path)?;
            return Ok(audio_path);
        }

        Err(anyhow::anyhow!(
            "No audio file found in {}",
            recording_dir.display()
        ))
    }
}
