//! Transcription functionality for Scriba.
//!
//! Uses sherpa-onnx for local transcription (Whisper ONNX, SenseVoice, etc.)
//! and the OpenAI API for cloud transcription.

use anyhow::{Context, Result};
use futures_util::StreamExt;
use reqwest::{
    multipart::{Form, Part},
    Client,
};
use serde_json::Value;
use sherpa_onnx::{OfflineRecognizer, OfflineRecognizerConfig, Wave};
use std::io::{stdout, Write};
use std::path::{Path, PathBuf};
use std::io::BufReader;
use std::process::Command;
use std::time::{Duration, Instant};
use tokio::time::sleep;

use super::config::{LocalModel, ScribaConfig, TranscriptionMode};
use super::files::FileManager;
use crate::database::Database;
use crate::utils::BASE_PATH;

/// OpenAI Whisper API maximum upload size (25 MB).
const OPENAI_MAX_FILE_SIZE: u64 = 25 * 1024 * 1024;

/// Progress indicator for transcription operations.
pub struct TranscriptionProgress {
    start_time: Instant,
    animation_frame: usize,
}

impl TranscriptionProgress {
    pub fn new() -> Self {
        Self {
            start_time: Instant::now(),
            animation_frame: 0,
        }
    }

    pub async fn show_progress(&mut self, mode_message: Option<&str>) {
        let elapsed = self.start_time.elapsed().as_secs();

        let spinner_chars = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
        let spinner = spinner_chars[self.animation_frame % spinner_chars.len()];

        let message = match elapsed {
            0..=3 => "Preparing audio",
            4..=8 => "Processing",
            9..=25 => "Transcribing",
            _ => "Almost there, hang tight",
        };

        let time_display = if elapsed < 60 {
            format!("{}s", elapsed)
        } else {
            format!("{}m {}s", elapsed / 60, elapsed % 60)
        };

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
        print!(
            "\r🎵 {} [{}] {} - {}",
            spinner, bar_str, display_message, time_display
        );
        stdout().flush().unwrap();

        self.animation_frame += 1;
        sleep(Duration::from_millis(100)).await;
    }
}

impl Default for TranscriptionProgress {
    fn default() -> Self {
        Self::new()
    }
}

/// Persist transcript to file and update database.
fn save_transcript_to_files_and_db(
    audio_path: &Path,
    transcript_text: &str,
    model_used: &str,
) -> Result<()> {
    let audio_dir = audio_path
        .parent()
        .context("Could not determine audio file directory")?;
    let transcript_file_path = audio_dir.join("transcript.txt");
    std::fs::write(&transcript_file_path, transcript_text).with_context(|| {
        format!(
            "Failed to write transcript to {}",
            transcript_file_path.display()
        )
    })?;

    let mut db = Database::new().context("Failed to connect to database")?;
    let directory_name = audio_dir
        .file_name()
        .and_then(|name| name.to_str())
        .context("Could not determine directory name")?;
    if let Some(recording) = db.get_recording_by_directory(directory_name)? {
        if let Some(recording_id) = recording.id {
            db.upsert_transcript(recording_id, transcript_text)?;
            let _ = db.update_recording_transcript_status_and_model(
                recording_id,
                "completed",
                true,
                model_used,
            );
        }
    }

    Ok(())
}

pub(crate) fn find_ffmpeg() -> Result<String> {
    let possible_paths = [
        "ffmpeg",
        "/opt/homebrew/bin/ffmpeg",
        "/usr/bin/ffmpeg",
        "/usr/local/bin/ffmpeg",
        "C:\\ffmpeg\\bin\\ffmpeg.exe",
    ];

    for path in &possible_paths {
        match Command::new(path).arg("-version").output() {
            Ok(output) => {
                if output.status.success() {
                    return Ok(path.to_string());
                }
            }
            Err(_) => continue,
        }
    }

    match Command::new("ffmpeg").arg("-version").output() {
        Ok(output) if output.status.success() => {
            return Ok("ffmpeg".to_string());
        }
        _ => {}
    }

    Err(anyhow::anyhow!(
        "FFmpeg not found. Please install FFmpeg and ensure it's in your PATH."
    ))
}

fn ensure_mono_16k_wav(input: &Path) -> Result<PathBuf> {
    let out = input
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join("_tmp_stt_16k.wav");

    let ffmpeg_path = find_ffmpeg().context("FFmpeg is required for audio processing")?;

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

/// Get the duration of an audio file in seconds using ffprobe.
fn get_audio_duration_secs(audio_path: &Path) -> Result<f64> {
    let ffmpeg_path = find_ffmpeg()?;
    // Derive ffprobe path from ffmpeg path
    let ffprobe_path = ffmpeg_path.replace("ffmpeg", "ffprobe");

    let output = Command::new(&ffprobe_path)
        .args([
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "csv=p=0",
            audio_path.to_string_lossy().as_ref(),
        ])
        .output()
        .with_context(|| format!("Failed to run ffprobe from path: {}", ffprobe_path))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(anyhow::anyhow!("ffprobe failed: {}", stderr));
    }

    let duration_str = String::from_utf8_lossy(&output.stdout);
    duration_str
        .trim()
        .parse::<f64>()
        .context("Failed to parse audio duration from ffprobe output")
}

/// Split an audio file into chunks that each fit under `OPENAI_MAX_FILE_SIZE`.
///
/// Returns a list of temporary chunk file paths, sorted in order.
fn split_audio_into_chunks(audio_path: &Path) -> Result<Vec<PathBuf>> {
    let file_size = std::fs::metadata(audio_path)
        .context("Failed to read audio file metadata")?
        .len();

    let duration_secs = get_audio_duration_secs(audio_path)?;
    if duration_secs <= 0.0 {
        return Err(anyhow::anyhow!("Audio file has zero or negative duration"));
    }

    // Calculate how many chunks we need, with a 20% safety margin
    let target_chunk_size = OPENAI_MAX_FILE_SIZE as f64 * 0.80;
    let num_chunks = (file_size as f64 / target_chunk_size).ceil() as usize;
    let chunk_duration = duration_secs / num_chunks as f64;

    let ffmpeg_path = find_ffmpeg()?;
    let tmp_dir = audio_path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join("_tmp_chunks");
    std::fs::create_dir_all(&tmp_dir).context("Failed to create temp chunk directory")?;

    let extension = audio_path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("mp3");

    let mut chunk_paths = Vec::with_capacity(num_chunks);

    for i in 0..num_chunks {
        let start = i as f64 * chunk_duration;
        let chunk_path = tmp_dir.join(format!("chunk_{:04}.{}", i, extension));

        let output = Command::new(&ffmpeg_path)
            .args([
                "-y",
                "-i",
                audio_path.to_string_lossy().as_ref(),
                "-ss",
                &format!("{:.3}", start),
                "-t",
                &format!("{:.3}", chunk_duration),
                "-c",
                "copy",
                chunk_path.to_string_lossy().as_ref(),
            ])
            .output()
            .with_context(|| format!("Failed to split audio chunk {}", i))?;

        if !output.status.success() {
            // Clean up on failure
            let _ = std::fs::remove_dir_all(&tmp_dir);
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(anyhow::anyhow!("ffmpeg chunk split failed: {}", stderr));
        }

        chunk_paths.push(chunk_path);
    }

    Ok(chunk_paths)
}

/// Transcribe a single audio chunk via the OpenAI API.
async fn transcribe_single_chunk(audio_path: &Path, api_key: &str) -> Result<String> {
    let audio_file = std::fs::read(audio_path).context("Unable to read audio chunk")?;

    let filename = audio_path
        .file_name()
        .and_then(|f| f.to_str())
        .unwrap_or("audio")
        .to_string();

    let client = Client::new();

    let part = Part::bytes(audio_file)
        .file_name(filename)
        .mime_str("audio/mpeg")
        .context("Failed to create multipart form data")?;

    let form = Form::new().part("file", part).text("model", "whisper-1");

    let response = client
        .post("https://api.openai.com/v1/audio/transcriptions")
        .header("Authorization", format!("Bearer {}", api_key))
        .multipart(form)
        .send()
        .await
        .context("Failed to send transcription request to OpenAI")?;

    let status = response.status();
    if !status.is_success() {
        let error_text = response
            .text()
            .await
            .unwrap_or_else(|_| "Unknown error".to_string());
        return Err(anyhow::anyhow!(
            "OpenAI API request failed with status {}: {}",
            status,
            error_text
        ));
    }

    let response_json: Value = response
        .json()
        .await
        .context("Failed to parse OpenAI response as JSON")?;

    response_json
        .get("text")
        .and_then(|t| t.as_str())
        .map(|s| s.to_string())
        .ok_or_else(|| anyhow::anyhow!("No 'text' field found in OpenAI response"))
}

/// Paths to the files composing a sherpa-onnx model.
pub(crate) struct SherpaModelPaths {
    /// Directory containing the model files.
    pub dir: PathBuf,
}

/// Model archive info for downloading from sherpa-onnx releases.
struct ModelArchiveInfo {
    /// Archive filename (e.g. "sherpa-onnx-whisper-tiny.tar.bz2").
    archive_name: &'static str,
    /// Base URL for downloads.
    url: &'static str,
    /// Expected directory name inside the archive after extraction.
    extracted_dir: &'static str,
}

/// Extract a `.tar.bz2` archive to a destination directory using pure Rust.
/// Cross-platform: does not shell out to `tar`.
fn extract_tar_bz2(archive_path: &Path, dest_dir: &Path) -> Result<()> {
    let file = std::fs::File::open(archive_path)
        .with_context(|| format!("Failed to open archive: {}", archive_path.display()))?;
    let decoder = bzip2::read::BzDecoder::new(BufReader::new(file));
    let mut archive = tar::Archive::new(decoder);
    archive.unpack(dest_dir)
        .with_context(|| format!("Failed to extract archive to {}", dest_dir.display()))?;
    Ok(())
}

fn model_archive_info(model: LocalModel) -> ModelArchiveInfo {
    match model {
        LocalModel::WhisperTiny => ModelArchiveInfo {
            archive_name: "sherpa-onnx-whisper-tiny.tar.bz2",
            url: "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-tiny.tar.bz2",
            extracted_dir: "sherpa-onnx-whisper-tiny",
        },
        LocalModel::WhisperBase => ModelArchiveInfo {
            archive_name: "sherpa-onnx-whisper-base.tar.bz2",
            url: "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-base.tar.bz2",
            extracted_dir: "sherpa-onnx-whisper-base",
        },
        LocalModel::WhisperSmall => ModelArchiveInfo {
            archive_name: "sherpa-onnx-whisper-small.tar.bz2",
            url: "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-small.tar.bz2",
            extracted_dir: "sherpa-onnx-whisper-small",
        },
        LocalModel::WhisperMedium => ModelArchiveInfo {
            archive_name: "sherpa-onnx-whisper-medium.tar.bz2",
            url: "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-medium.tar.bz2",
            extracted_dir: "sherpa-onnx-whisper-medium",
        },
        LocalModel::WhisperLarge => ModelArchiveInfo {
            archive_name: "sherpa-onnx-whisper-large-v3.tar.bz2",
            url: "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-large-v3.tar.bz2",
            extracted_dir: "sherpa-onnx-whisper-large-v3",
        },
        LocalModel::WhisperTurbo => ModelArchiveInfo {
            archive_name: "sherpa-onnx-whisper-turbo.tar.bz2",
            url: "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-turbo.tar.bz2",
            extracted_dir: "sherpa-onnx-whisper-turbo",
        },
        LocalModel::SenseVoice => ModelArchiveInfo {
            archive_name: "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2",
            url: "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2",
            extracted_dir: "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17",
        },
        LocalModel::ParakeetTdt => ModelArchiveInfo {
            archive_name: "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8.tar.bz2",
            url: "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8.tar.bz2",
            extracted_dir: "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8",
        },
    }
}

/// Check if the ONNX model is already downloaded locally.
pub(crate) fn check_model_downloaded(model: LocalModel) -> bool {
    let info = model_archive_info(model);
    let model_dir = BASE_PATH.join("models").join("sherpa").join(info.extracted_dir);
    model_dir.exists() && model_dir.is_dir()
}

/// Ensure the ONNX model is downloaded and return its paths.
pub(crate) async fn ensure_sherpa_model(model: LocalModel, quiet: bool) -> Result<SherpaModelPaths> {
    let models_dir = BASE_PATH.join("models").join("sherpa");
    std::fs::create_dir_all(&models_dir).ok();

    let info = model_archive_info(model);
    let model_dir = models_dir.join(info.extracted_dir);

    if model_dir.exists() {
        return Ok(SherpaModelPaths { dir: model_dir });
    }

    if !quiet {
        println!(
            "Downloading {} model (this may take a while)...",
            model.display_name()
        );
    }

    // Download and extract the tarball
    let archive_path = models_dir.join(info.archive_name);
    download_file_streaming(info.url, &archive_path, quiet)
        .await
        .with_context(|| format!("Failed to download model from {}", info.url))?;

    // Extract tar.bz2 using pure Rust (cross-platform)
    if let Err(e) = extract_tar_bz2(&archive_path, &models_dir) {
        let _ = std::fs::remove_file(&archive_path);
        return Err(e);
    }
    let _ = std::fs::remove_file(&archive_path);

    if !quiet {
        println!("Model downloaded to {}", model_dir.display());
    }

    Ok(SherpaModelPaths { dir: model_dir })
}

/// Download a model, sending progress (0-100) through the channel.
pub(crate) async fn download_model_with_progress(
    model: LocalModel,
    tx: tokio::sync::mpsc::UnboundedSender<u8>,
) -> Result<()> {
    let models_dir = BASE_PATH.join("models").join("sherpa");
    std::fs::create_dir_all(&models_dir).ok();

    if check_model_downloaded(model) {
        let _ = tx.send(100);
        return Ok(());
    }

    let info = model_archive_info(model);
    let archive_path = models_dir.join(info.archive_name);

    let client = Client::new();
    let resp = client.get(info.url).send().await?.error_for_status()?;
    let total = resp.content_length();
    let mut stream = resp.bytes_stream();
    let mut file = std::fs::File::create(&archive_path).context("Failed to create model file")?;
    let mut downloaded: u64 = 0;

    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        std::io::Write::write_all(&mut file, &chunk)?;
        downloaded += chunk.len() as u64;
        if let Some(total) = total {
            // Reserve last 5% for extraction
            let pct = ((downloaded as f64 / total as f64) * 95.0).min(95.0) as u8;
            let _ = tx.send(pct);
        }
    }

    // Extract (pure Rust, cross-platform)
    if let Err(e) = extract_tar_bz2(&archive_path, &models_dir) {
        let _ = std::fs::remove_file(&archive_path);
        return Err(e);
    }
    let _ = std::fs::remove_file(&archive_path);

    let _ = tx.send(100);
    Ok(())
}

async fn download_file_streaming(url: &str, dest: &Path, quiet: bool) -> Result<()> {
    let client = Client::new();
    let resp = client.get(url).send().await?.error_for_status()?;
    let total = resp.content_length();
    let mut stream = resp.bytes_stream();
    let mut file = std::fs::File::create(dest).context("Failed to create destination file")?;
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
    if !quiet {
        println!();
    }
    Ok(())
}

/// Get the file name prefix used by sherpa-onnx Whisper model archives.
/// e.g., WhisperTiny -> "tiny", WhisperLarge -> "large-v3"
///
/// Only valid for Whisper-family models. Non-Whisper models use their own naming.
fn whisper_file_prefix(model: LocalModel) -> &'static str {
    match model {
        LocalModel::WhisperTiny => "tiny",
        LocalModel::WhisperBase => "base",
        LocalModel::WhisperSmall => "small",
        LocalModel::WhisperMedium => "medium",
        LocalModel::WhisperLarge => "large-v3",
        LocalModel::WhisperTurbo => "turbo",
        _ => unreachable!("whisper_file_prefix called for non-Whisper model"),
    }
}

/// Build the sherpa-onnx recognizer config for a given model.
pub(crate) fn build_recognizer_config(model: LocalModel, model_dir: &Path) -> OfflineRecognizerConfig {
    let mut config = OfflineRecognizerConfig::default();
    config.model_config.num_threads = std::thread::available_parallelism()
        .map(|n| n.get() as i32)
        .unwrap_or(4);

    match model {
        LocalModel::WhisperTiny
        | LocalModel::WhisperBase
        | LocalModel::WhisperSmall
        | LocalModel::WhisperMedium
        | LocalModel::WhisperLarge
        | LocalModel::WhisperTurbo => {
            let prefix = whisper_file_prefix(model);
            // sherpa-onnx whisper archives use: {prefix}-encoder[.int8].onnx
            // Prefer non-quantized, fall back to int8 (some archives only have int8)
            let encoder = if model_dir.join(format!("{}-encoder.onnx", prefix)).exists() {
                model_dir.join(format!("{}-encoder.onnx", prefix))
            } else {
                model_dir.join(format!("{}-encoder.int8.onnx", prefix))
            };
            let decoder = if model_dir.join(format!("{}-decoder.onnx", prefix)).exists() {
                model_dir.join(format!("{}-decoder.onnx", prefix))
            } else {
                model_dir.join(format!("{}-decoder.int8.onnx", prefix))
            };
            config.model_config.whisper.encoder =
                Some(encoder.to_string_lossy().into_owned());
            config.model_config.whisper.decoder =
                Some(decoder.to_string_lossy().into_owned());
            config.model_config.tokens =
                Some(model_dir.join(format!("{}-tokens.txt", prefix)).to_string_lossy().into_owned());
            // No language set → auto-detect (Scriba is language-agnostic)
            config.model_config.whisper.task = Some("transcribe".into());
        }
        LocalModel::SenseVoice => {
            // SenseVoice archives ship both model.onnx and model.int8.onnx.
            // Prefer int8: it's the recommended variant (faster, similar quality).
            let model_file = if model_dir.join("model.int8.onnx").exists() {
                "model.int8.onnx"
            } else {
                "model.onnx"
            };
            config.model_config.sense_voice.model =
                Some(model_dir.join(model_file).to_string_lossy().into_owned());
            config.model_config.sense_voice.language = Some("auto".into());
            config.model_config.sense_voice.use_itn = true;
            config.model_config.tokens =
                Some(model_dir.join("tokens.txt").to_string_lossy().into_owned());
        }
        LocalModel::ParakeetTdt => {
            // Transducer architecture: encoder + decoder + joiner
            // Prefer int8: this archive only ships int8 variants.
            let encoder = if model_dir.join("encoder.int8.onnx").exists() {
                "encoder.int8.onnx"
            } else {
                "encoder.onnx"
            };
            let decoder = if model_dir.join("decoder.int8.onnx").exists() {
                "decoder.int8.onnx"
            } else {
                "decoder.onnx"
            };
            let joiner = if model_dir.join("joiner.int8.onnx").exists() {
                "joiner.int8.onnx"
            } else {
                "joiner.onnx"
            };
            config.model_config.transducer.encoder =
                Some(model_dir.join(encoder).to_string_lossy().into_owned());
            config.model_config.transducer.decoder =
                Some(model_dir.join(decoder).to_string_lossy().into_owned());
            config.model_config.transducer.joiner =
                Some(model_dir.join(joiner).to_string_lossy().into_owned());
            config.model_config.tokens =
                Some(model_dir.join("tokens.txt").to_string_lossy().into_owned());
            config.model_config.model_type = Some("nemo_transducer".into());
        }
    }

    config
}

/// Ensure the Silero VAD model is downloaded and return its path.
async fn ensure_vad_model() -> Result<PathBuf> {
    let vad_dir = BASE_PATH.join("models").join("sherpa");
    std::fs::create_dir_all(&vad_dir).ok();
    let vad_path = vad_dir.join("silero_vad.onnx");
    if !vad_path.exists() {
        let url = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx";
        download_file_streaming(url, &vad_path, true).await
            .context("Failed to download Silero VAD model")?;
    }
    Ok(vad_path)
}

/// Create a Silero VAD for segmenting long audio.
fn create_vad(vad_model_path: &Path) -> Result<sherpa_onnx::VoiceActivityDetector> {
    let mut vad_config = sherpa_onnx::VadModelConfig::default();
    vad_config.silero_vad.model = Some(vad_model_path.to_string_lossy().into_owned());
    vad_config.silero_vad.threshold = 0.5;
    vad_config.silero_vad.min_silence_duration = 0.5;
    vad_config.silero_vad.min_speech_duration = 0.25;
    vad_config.silero_vad.window_size = 512;
    vad_config.sample_rate = 16000;
    vad_config.num_threads = 1;

    // Buffer 30 seconds — we feed audio in small chunks and drain segments as they appear
    sherpa_onnx::VoiceActivityDetector::create(&vad_config, 30.0)
        .ok_or_else(|| anyhow::anyhow!("Failed to create Silero VAD"))
}

/// Suppress sherpa-onnx C library stderr output (e.g. "Only waves less than 30 seconds").
/// Returns a guard that restores stderr on drop.
///
/// NOTE: This redirects stderr process-wide. Any other thread writing to stderr
/// while the guard is held will have its output silently discarded. The guard is
/// short-lived (only during the sherpa-onnx transcription call).
fn suppress_stderr() -> Option<gag::Hold> {
    gag::Hold::stderr().ok()
}


/// Run transcription using sherpa-onnx with VAD segmentation for long audio.
fn run_sherpa_transcription(model: LocalModel, model_dir: &Path, wav_path: &Path, vad_model_path: &Path) -> Result<String> {
    let config = build_recognizer_config(model, model_dir);

    let recognizer = OfflineRecognizer::create(&config)
        .ok_or_else(|| anyhow::anyhow!("Failed to create sherpa-onnx recognizer. Check model files in {}", model_dir.display()))?;

    let wave = Wave::read(wav_path.to_string_lossy().as_ref())
        .ok_or_else(|| anyhow::anyhow!("Failed to read WAV file: {}", wav_path.display()))?;

    let samples = wave.samples();
    let sample_rate = wave.sample_rate();

    let vad = create_vad(vad_model_path)?;
    let window_size = 512; // Silero VAD window size
    let mut all_text = String::new();

    let drain_text = |vad: &sherpa_onnx::VoiceActivityDetector,
                          recognizer: &OfflineRecognizer,
                          sample_rate: i32,
                          all_text: &mut String| {
        while !vad.is_empty() {
            if let Some(segment) = vad.front() {
                let stream = recognizer.create_stream();
                stream.accept_waveform(sample_rate, segment.samples());
                recognizer.decode(&stream);
                if let Some(result) = stream.get_result() {
                    let text = result.text.trim();
                    if !text.is_empty() {
                        if !all_text.is_empty() {
                            all_text.push(' ');
                        }
                        all_text.push_str(text);
                    }
                }
            }
            vad.pop();
        }
    };

    for chunk in samples.chunks(window_size) {
        vad.accept_waveform(chunk);
        drain_text(&vad, &recognizer, sample_rate, &mut all_text);
    }

    vad.flush();
    drain_text(&vad, &recognizer, sample_rate, &mut all_text);

    Ok(all_text)
}

async fn transcribe_with_openai_api(audio_path: &PathBuf, api_key: &str) -> Result<String> {
    let file_size = std::fs::metadata(audio_path)
        .context("Failed to read audio file metadata")?
        .len();

    if file_size <= OPENAI_MAX_FILE_SIZE {
        return transcribe_single_chunk(audio_path, api_key).await;
    }

    // File exceeds 25MB — split into chunks, transcribe each, concatenate
    let chunk_paths = split_audio_into_chunks(audio_path)?;
    let num_chunks = chunk_paths.len();
    let mut transcripts = Vec::with_capacity(num_chunks);

    for (i, chunk_path) in chunk_paths.iter().enumerate() {
        let text = transcribe_single_chunk(chunk_path, api_key)
            .await
            .with_context(|| format!("Failed to transcribe chunk {}/{}", i + 1, num_chunks))?;

        transcripts.push(text);
    }

    // Clean up temp chunk files
    if let Some(tmp_dir) = chunk_paths.first().and_then(|p| p.parent()) {
        let _ = std::fs::remove_dir_all(tmp_dir);
    }

    Ok(transcripts.join(" "))
}

/// Unified transcription function.
pub async fn transcribe_audio(
    input_path: &PathBuf,
    mode_override: Option<TranscriptionMode>,
    verbose: bool,
) -> Result<()> {
    let audio_file_path = FileManager::resolve_audio_path(input_path)?;
    let config = ScribaConfig::load()?;
    let transcription_mode = mode_override.unwrap_or_else(|| config.transcription.clone());

    let progress = TranscriptionProgress::new();

    if verbose {
        let mode_description = match &transcription_mode {
            TranscriptionMode::Local { model } => {
                format!(
                    "Transcribing locally using {} model...",
                    model.display_name()
                )
            }
            TranscriptionMode::Api { .. } => {
                "Transcribing using OpenAI Whisper API...".to_string()
            }
        };
        println!("\n{}\n", mode_description);
    }

    let (transcription_text, model_used) = match transcription_mode {
        TranscriptionMode::Local { model } => {
            // Suppress sherpa-onnx C library stderr warnings that would corrupt the TUI
            let _stderr_guard = suppress_stderr();

            let progress_task = if verbose {
                let mut local_progress = progress;
                Some(tokio::spawn(async move {
                    loop {
                        let message = match local_progress.start_time.elapsed().as_secs() {
                            0..=3 => Some("Preparing audio (16kHz mono)"),
                            4..=8 => Some("Loading model"),
                            9..=25 => Some("Running local transcription"),
                            _ => Some("Almost there, hang tight"),
                        };
                        local_progress.show_progress(message).await;
                    }
                }))
            } else {
                None
            };

            let wav_path = ensure_mono_16k_wav(&audio_file_path)
                .context("Failed to prepare 16kHz mono WAV for transcription")?;

            let model_paths = {
                let download_future = ensure_sherpa_model(model, true);
                tokio::time::timeout(Duration::from_secs(600), download_future)
                    .await
                    .context("Model download timed out after 10 minutes")?
                    .context("Failed to download model")?
            };

            // Ensure VAD model is available (small ~2MB download)
            let vad_model_path = ensure_vad_model().await
                .context("Failed to download VAD model")?;

            let text = run_sherpa_transcription(model, &model_paths.dir, &wav_path, &vad_model_path)
                .context("Local transcription failed")?;

            if wav_path.file_name() == Some(std::ffi::OsStr::new("_tmp_stt_16k.wav")) {
                let _ = std::fs::remove_file(&wav_path);
            }
            if let Some(task) = progress_task {
                task.abort();
            }
            let model_name = format!("sherpa-{}", model);
            (text, model_name)
        }
        TranscriptionMode::Api { api_key } => {
            let progress_task = if verbose {
                let mut api_progress = progress;
                Some(tokio::spawn(async move {
                    loop {
                        let message = match api_progress.start_time.elapsed().as_secs() {
                            0..=3 => Some("Uploading audio file"),
                            4..=15 => Some("OpenAI is processing your audio"),
                            16..=30 => Some("Converting speech to text"),
                            31..=60 => Some("Transcribing (large files are split into chunks)"),
                            _ => Some("Still transcribing, hang tight"),
                        };
                        api_progress.show_progress(message).await;
                    }
                }))
            } else {
                None
            };

            let result = transcribe_with_openai_api(&audio_file_path, &api_key)
                .await
                .context("OpenAI API transcription failed")?;
            if let Some(task) = progress_task {
                task.abort();
            }
            (result, "whisper-1".to_string())
        }
    };

    if verbose {
        print!("\r{}", " ".repeat(80));
        print!("\r");
        stdout().flush().unwrap();
        println!("Transcription complete!");
    }

    save_transcript_to_files_and_db(
        &audio_file_path,
        &transcription_text,
        &model_used,
    )?;

    if verbose {
        let transcript_file_path = audio_file_path
            .parent()
            .context("Could not determine audio file directory")?
            .join("transcript.txt");
        println!(
            "\nTranscript saved to: {}",
            transcript_file_path.display()
        );
    }

    Ok(())
}

// Whisper initial-prompt functions were removed during the sherpa-onnx migration.
// sherpa-onnx's Whisper bindings do not currently support initial prompting.
// If support is added upstream, context-priming can be reintroduced from git history.
