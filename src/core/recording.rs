//! Audio recording functionality for Scriba.

use anyhow::{Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::signal;
use tokio::sync::mpsc;

use super::audio::{
    convert_wav_to_mp3, create_encoder, AudioEncoder, AudioFormat, CompressionSettings,
};
use super::files::FileManager;
use crate::database::{Database, Recording};
use crate::utils::BASE_PATH;
use chrono::Utc;

/// Monitors audio input levels for real-time feedback during recording.
pub struct AudioLevelMonitor {
    last_update: Instant,
    level: f32,
}

impl AudioLevelMonitor {
    pub fn new() -> Self {
        Self {
            last_update: Instant::now(),
            level: 0.0,
        }
    }

    /// Update the audio level with new samples.
    /// Returns the current level if enough time has passed for UI updates.
    pub fn update_level(&mut self, samples: &[f32]) -> Option<f32> {
        let sum_of_squares: f32 = samples.iter().map(|&s| s * s).sum();
        let rms = (sum_of_squares / samples.len() as f32).sqrt();

        self.level = self.level * 0.8 + rms * 0.2;

        if self.last_update.elapsed() >= Duration::from_millis(100) {
            self.last_update = Instant::now();
            Some(self.level)
        } else {
            None
        }
    }
}

impl Default for AudioLevelMonitor {
    fn default() -> Self {
        Self::new()
    }
}

/// Unified recording options controlling behavior and integrations.
pub struct RecordOptions {
    pub compression_settings: Option<CompressionSettings>,
    pub stop_rx: Option<mpsc::Receiver<()>>,
    pub level_tx: Option<mpsc::Sender<f32>>,
    pub verbose: bool,
}

impl Default for RecordOptions {
    fn default() -> Self {
        Self {
            compression_settings: None,
            stop_rx: None,
            level_tx: None,
            verbose: false,
        }
    }
}

type AudioEncoderHandle = Arc<Mutex<Option<Box<dyn AudioEncoder>>>>;

/// Get file paths and format info for recording based on compression settings.
fn setup_recording_paths(
    output_path: &PathBuf,
    compression_settings: &Option<CompressionSettings>,
) -> (PathBuf, PathBuf, String, bool) {
    let wav_file_path = BASE_PATH.join(output_path).join("recording.wav");

    let (final_file_path, audio_format_str, needs_conversion) =
        if let Some(ref settings) = compression_settings {
            match settings.format {
                AudioFormat::Mp3 => {
                    let mp3_path = BASE_PATH.join(output_path).join("recording.mp3");
                    (mp3_path, "mp3".to_string(), true)
                }
                _ => {
                    let filename = settings.get_filename("recording");
                    (
                        BASE_PATH.join(output_path).join(filename),
                        settings.format.to_string().to_lowercase(),
                        false,
                    )
                }
            }
        } else {
            (wav_file_path.clone(), "wav".to_string(), false)
        };

    (
        wav_file_path,
        final_file_path,
        audio_format_str,
        needs_conversion,
    )
}

/// Create a database recording entry using unified metadata extraction.
fn create_recording_entry(
    output_path: &PathBuf,
    final_file_path: &PathBuf,
) -> Result<Recording, anyhow::Error> {
    let meta = FileManager::extract_audio_metadata(final_file_path)?;

    Ok(Recording {
        id: None,
        directory_name: output_path.to_string_lossy().to_string(),
        display_name: None,
        created_at: Utc::now(),
        updated_at: Utc::now(),
        duration_seconds: meta.duration_seconds,
        file_size_bytes: meta.file_size_bytes,
        audio_format: meta.audio_format,
        sample_rate: meta.sample_rate,
        channels: meta.channels,
        has_transcript: false,
        transcript_status: "pending".to_string(),
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
        audio_path: final_file_path
            .file_name()
            .unwrap()
            .to_string_lossy()
            .to_string(),
        transcript_path: None,
    })
}

/// Core recording function.
fn record_core(
    output_path: PathBuf,
    compression_settings: Option<CompressionSettings>,
    level_tx_opt: Option<mpsc::Sender<f32>>,
    verbose: bool,
    wait_stop: Box<dyn FnOnce()>,
) -> Result<String, anyhow::Error> {
    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .ok_or_else(|| anyhow::anyhow!("No input device available"))?;

    let config = device
        .default_input_config()
        .map_err(|e| anyhow::anyhow!("Failed to get default input config: {}", e))?;

    let (wav_file_path, final_file_path, _audio_format_str, needs_conversion) =
        setup_recording_paths(&output_path, &compression_settings);

    if let Some(parent) = wav_file_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let recording_settings = CompressionSettings {
        format: AudioFormat::Wav,
        sample_rate: config.sample_rate().0,
        bitrate_kbps: None,
        channels: config.channels(),
        speech_optimized: false,
    };
    let encoder = create_encoder(&wav_file_path, &recording_settings)?;
    let encoder = Arc::new(Mutex::new(Some(encoder)));

    let level_monitor = Arc::new(Mutex::new(AudioLevelMonitor::new()));

    if verbose {
        println!(
            "🎙️  Begin recording... (Press {} to stop)",
            if level_tx_opt.is_some() {
                "Esc in UI"
            } else {
                "Ctrl+C"
            }
        );
        println!("Device config: {:?}", config);
    }

    let encoder_2 = encoder.clone();
    let level_monitor_2 = level_monitor.clone();

    let err_fn = move |err| {
        eprintln!("an error occurred on stream: {}", err);
    };

    let stream = match config.sample_format() {
        cpal::SampleFormat::I8 => device.build_input_stream(
            &config.clone().into(),
            move |data: &[i8], _: &_| {
                write_input_data_i8(data, &encoder_2, &level_monitor_2, level_tx_opt.as_ref())
            },
            err_fn,
            None,
        )?,
        cpal::SampleFormat::I16 => device.build_input_stream(
            &config.clone().into(),
            move |data: &[i16], _: &_| {
                write_input_data_i16(data, &encoder_2, &level_monitor_2, level_tx_opt.as_ref())
            },
            err_fn,
            None,
        )?,
        cpal::SampleFormat::I32 => device.build_input_stream(
            &config.clone().into(),
            move |data: &[i32], _: &_| {
                write_input_data_i32(data, &encoder_2, &level_monitor_2, level_tx_opt.as_ref())
            },
            err_fn,
            None,
        )?,
        cpal::SampleFormat::F32 => device.build_input_stream(
            &config.clone().into(),
            move |data: &[f32], _: &_| {
                write_input_data_f32(data, &encoder_2, &level_monitor_2, level_tx_opt.as_ref())
            },
            err_fn,
            None,
        )?,
        sample_format => {
            return Err(anyhow::Error::msg(format!(
                "Unsupported sample format '{sample_format}'"
            )))
        }
    };

    stream.play()?;
    wait_stop();
    drop(stream);

    if let Some(mut enc) = encoder.lock().unwrap().take() {
        enc.finalize()?;
    }

    if needs_conversion {
        if let Some(ref settings) = compression_settings {
            convert_wav_to_mp3(&wav_file_path, &final_file_path, settings)
                .context("Failed to convert WAV to MP3")?;
            let _ = std::fs::remove_file(&wav_file_path);
        }
    }

    let metadata_file_path = if needs_conversion {
        &final_file_path
    } else {
        &wav_file_path
    };

    let mut db = Database::new().context("Failed to connect to database")?;
    let recording = create_recording_entry(&output_path, metadata_file_path)?;

    match db.insert_recording(&recording) {
        Ok(id) => {
            if verbose {
                println!("📊 Recording saved to database with ID: {}", id);
            }
        }
        Err(e) => eprintln!("⚠️ Warning: Failed to save recording to database: {}", e),
    }

    if verbose {
        println!("✅ Recording complete: {}", metadata_file_path.display());
    }

    Ok(output_path.to_string_lossy().to_string())
}

/// Record audio from the default input device.
pub async fn record_audio(output_path: PathBuf, options: RecordOptions) -> Result<String> {
    let wait_strategy: Box<dyn FnOnce()> = match options.stop_rx {
        Some(mut rx) => Box::new(move || {
            let (tx, rx_done) = std::sync::mpsc::channel();
            std::thread::spawn(move || {
                let _ = rx.blocking_recv();
                let _ = tx.send(());
            });
            let _ = rx_done.recv();
        }),
        None => {
            let handle = tokio::runtime::Handle::current();
            Box::new(move || {
                let (tx, rx) = std::sync::mpsc::channel();
                let handle_clone = handle.clone();
                std::thread::spawn(move || {
                    let _ = handle_clone.block_on(signal::ctrl_c());
                    let _ = tx.send(());
                });
                let _ = rx.recv();
            })
        }
    };

    record_core(
        output_path,
        options.compression_settings,
        options.level_tx,
        options.verbose,
        wait_strategy,
    )
}

// Sample handlers
fn write_input_data_f32(
    input: &[f32],
    encoder: &AudioEncoderHandle,
    level_monitor: &Arc<Mutex<AudioLevelMonitor>>,
    level_tx: Option<&mpsc::Sender<f32>>,
) {
    if let Ok(mut monitor) = level_monitor.try_lock() {
        if let Some(level) = monitor.update_level(input) {
            if let Some(tx) = level_tx {
                let _ = tx.try_send(level);
            }
        }
    }

    if let Ok(mut guard) = encoder.try_lock() {
        if let Some(encoder) = guard.as_mut() {
            let _ = encoder.encode_samples(input);
        }
    }
}

fn write_input_data_i16(
    input: &[i16],
    encoder: &AudioEncoderHandle,
    level_monitor: &Arc<Mutex<AudioLevelMonitor>>,
    level_tx: Option<&mpsc::Sender<f32>>,
) {
    let f32_samples: Vec<f32> = input.iter().map(|&s| s as f32 / i16::MAX as f32).collect();
    write_input_data_f32(&f32_samples, encoder, level_monitor, level_tx);
}

fn write_input_data_i32(
    input: &[i32],
    encoder: &AudioEncoderHandle,
    level_monitor: &Arc<Mutex<AudioLevelMonitor>>,
    level_tx: Option<&mpsc::Sender<f32>>,
) {
    let f32_samples: Vec<f32> = input.iter().map(|&s| s as f32 / i32::MAX as f32).collect();
    write_input_data_f32(&f32_samples, encoder, level_monitor, level_tx);
}

fn write_input_data_i8(
    input: &[i8],
    encoder: &AudioEncoderHandle,
    level_monitor: &Arc<Mutex<AudioLevelMonitor>>,
    level_tx: Option<&mpsc::Sender<f32>>,
) {
    let f32_samples: Vec<f32> = input.iter().map(|&s| s as f32 / i8::MAX as f32).collect();
    write_input_data_f32(&f32_samples, encoder, level_monitor, level_tx);
}
