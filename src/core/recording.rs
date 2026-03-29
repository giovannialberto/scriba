//! Audio recording functionality for Scriba.

use anyhow::{Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::signal;
use tokio::sync::mpsc;

use super::audio::{
    convert_wav_to_mp3, create_encoder, merge_wav_files, AudioEncoder, AudioFormat,
    CompressionSettings,
};
use super::files::FileManager;
use super::loopback;
use crate::database::{Database, Recording};
use crate::utils::BASE_PATH;
use chrono::Utc;

/// RMS level below which we consider the mic dead (near-zero signal).
/// Typical closed-lid / muted mic noise floor sits around 0.002-0.004.
const SILENCE_THRESHOLD: f32 = 0.005;

/// List all available audio input devices by name.
pub fn list_input_devices() -> Result<Vec<String>> {
    let host = cpal::default_host();
    let devices = host.input_devices()
        .map_err(|e| anyhow::anyhow!("Failed to enumerate input devices: {}", e))?;
    let names: Vec<String> = devices
        .filter_map(|d| d.name().ok())
        .collect();
    Ok(names)
}

/// Resolve an input device by name, falling back to the system default.
/// If `device_name` is `Some`, searches all input devices for a substring match.
pub fn resolve_input_device(device_name: Option<&str>) -> Result<cpal::Device> {
    let host = cpal::default_host();

    if let Some(name) = device_name {
        let devices = host.input_devices()
            .map_err(|e| anyhow::anyhow!("Failed to enumerate input devices: {}", e))?;
        let name_lower = name.to_lowercase();
        for device in devices {
            if let Ok(dev_name) = device.name() {
                if dev_name.to_lowercase().contains(&name_lower) {
                    return Ok(device);
                }
            }
        }
        // No match found — warn and fall back to default
        eprintln!(
            "⚠️  Input device '{}' not found, falling back to system default. \
             Use `scriba health --verbose` to list available devices.",
            name
        );
    }

    host.default_input_device()
        .ok_or_else(|| anyhow::anyhow!("No input device available"))
}

/// Monitors audio input levels for real-time feedback during recording.
pub struct AudioLevelMonitor {
    last_update: Instant,
    level: f32,
    silence_start: Option<Instant>,
}

impl AudioLevelMonitor {
    pub fn new() -> Self {
        Self {
            last_update: Instant::now(),
            level: 0.0,
            silence_start: None,
        }
    }

    /// Update the audio level with new samples.
    /// Returns the current level if enough time has passed for UI updates.
    pub fn update_level(&mut self, samples: &[f32]) -> Option<f32> {
        let sum_of_squares: f32 = samples.iter().map(|&s| s * s).sum();
        let rms = (sum_of_squares / samples.len() as f32).sqrt();

        self.level = self.level * 0.8 + rms * 0.2;

        // Update silence tracking
        if self.level < SILENCE_THRESHOLD {
            if self.silence_start.is_none() {
                self.silence_start = Some(Instant::now());
            }
        } else {
            self.silence_start = None;
        }

        if self.last_update.elapsed() >= Duration::from_millis(100) {
            self.last_update = Instant::now();
            Some(self.level)
        } else {
            None
        }
    }

    /// Returns how long continuous silence has lasted (Duration::ZERO if not silent).
    pub fn silence_duration(&self) -> Duration {
        self.silence_start
            .map(|start| start.elapsed())
            .unwrap_or(Duration::ZERO)
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
    /// If set, recording auto-stops after this duration of continuous silence.
    pub silence_timeout: Option<Duration>,
    /// Preferred input device name. Substring-matched against available devices.
    /// Falls back to system default if not found or not set.
    pub input_device: Option<String>,
    /// System audio loopback device name for capturing the other side of a call.
    /// On macOS uses ScreenCaptureKit natively; on Linux uses PulseAudio/PipeWire
    /// monitor sources. When set, both mic and system audio are recorded and merged.
    pub loopback_device: Option<String>,
}

impl Default for RecordOptions {
    fn default() -> Self {
        Self {
            compression_settings: None,
            stop_rx: None,
            level_tx: None,
            verbose: false,
            silence_timeout: None,
            input_device: None,
            loopback_device: None,
        }
    }
}

/// Result of a recording session.
pub struct RecordingResult {
    pub recording_name: String,
    pub auto_stopped: bool,
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
    silence_timeout: Option<Duration>,
    silence_flag: Arc<AtomicBool>,
    last_callback_ms: Arc<AtomicU64>,
    wait_stop: Box<dyn FnOnce()>,
    input_device_name: Option<String>,
    loopback_device_name: Option<String>,
) -> Result<RecordingResult, anyhow::Error> {
    let device = resolve_input_device(input_device_name.as_deref())?;

    let config = device
        .default_input_config()
        .map_err(|e| anyhow::anyhow!("Failed to get default input config: {}", e))?;

    let (wav_file_path, final_file_path, _audio_format_str, needs_conversion) =
        setup_recording_paths(&output_path, &compression_settings);

    if let Some(parent) = wav_file_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    // When loopback is enabled, mic goes to a temp file; otherwise straight to recording.wav
    let use_loopback = loopback_device_name.is_some();
    let mic_wav_path = if use_loopback {
        wav_file_path.with_file_name("mic_temp.wav")
    } else {
        wav_file_path.clone()
    };

    let recording_settings = CompressionSettings {
        format: AudioFormat::Wav,
        sample_rate: config.sample_rate().0,
        bitrate_kbps: None,
        channels: config.channels(),
        speech_optimized: false,
    };
    let encoder = create_encoder(&mic_wav_path, &recording_settings)?;
    let encoder = Arc::new(Mutex::new(Some(encoder)));

    // Start loopback capture if configured
    let loopback_wav_path = wav_file_path.with_file_name("loopback_temp.wav");
    let loopback_session = if use_loopback {
        match loopback::start_loopback_capture(
            loopback_device_name.as_deref(),
            &loopback_wav_path,
        ) {
            Ok(session) => {
                if verbose {
                    println!("🔊 Loopback capture started (system audio)");
                }
                Some(session)
            }
            Err(e) => {
                eprintln!(
                    "⚠️  Loopback capture failed, recording mic only: {}",
                    e
                );
                None
            }
        }
    } else {
        None
    };

    let level_monitor = Arc::new(Mutex::new(AudioLevelMonitor::new()));

    if verbose {
        let device_name = device.name().unwrap_or_else(|_| "Unknown".to_string());
        println!(
            "🎙️  Begin recording... (Press {} to stop)",
            if level_tx_opt.is_some() {
                "Esc in UI"
            } else {
                "Ctrl+C"
            }
        );
        println!("Input device: {}", device_name);
        println!("Device config: {:?}", config);
    }

    let encoder_2 = encoder.clone();
    let level_monitor_2 = level_monitor.clone();
    let silence_flag_2 = silence_flag.clone();
    let lcb = last_callback_ms.clone();

    let err_fn = move |err| {
        eprintln!("an error occurred on stream: {}", err);
    };

    let stream = match config.sample_format() {
        cpal::SampleFormat::I8 => {
            let sf = silence_flag_2.clone();
            let st = silence_timeout;
            let cb = lcb.clone();
            device.build_input_stream(
                &config.clone().into(),
                move |data: &[i8], _: &_| {
                    write_input_data_i8(data, &encoder_2, &level_monitor_2, level_tx_opt.as_ref(), &sf, st, &cb)
                },
                err_fn,
                None,
            )?
        }
        cpal::SampleFormat::I16 => {
            let sf = silence_flag_2.clone();
            let st = silence_timeout;
            let cb = lcb.clone();
            device.build_input_stream(
                &config.clone().into(),
                move |data: &[i16], _: &_| {
                    write_input_data_i16(data, &encoder_2, &level_monitor_2, level_tx_opt.as_ref(), &sf, st, &cb)
                },
                err_fn,
                None,
            )?
        }
        cpal::SampleFormat::I32 => {
            let sf = silence_flag_2.clone();
            let st = silence_timeout;
            let cb = lcb.clone();
            device.build_input_stream(
                &config.clone().into(),
                move |data: &[i32], _: &_| {
                    write_input_data_i32(data, &encoder_2, &level_monitor_2, level_tx_opt.as_ref(), &sf, st, &cb)
                },
                err_fn,
                None,
            )?
        }
        cpal::SampleFormat::F32 => {
            let sf = silence_flag_2.clone();
            let st = silence_timeout;
            let cb = lcb.clone();
            device.build_input_stream(
                &config.clone().into(),
                move |data: &[f32], _: &_| {
                    write_input_data_f32(data, &encoder_2, &level_monitor_2, level_tx_opt.as_ref(), &sf, st, &cb)
                },
                err_fn,
                None,
            )?
        }
        sample_format => {
            return Err(anyhow::Error::msg(format!(
                "Unsupported sample format '{sample_format}'"
            )))
        }
    };

    stream.play()?;
    wait_stop();
    let auto_stopped = silence_flag.load(Ordering::Relaxed);
    drop(stream);

    // Finalize mic encoder
    if let Some(mut enc) = encoder.lock().unwrap().take() {
        enc.finalize()?;
    }

    // Finalize loopback and merge with mic audio
    if let Some(session) = loopback_session {
        session
            .handle
            .stop_and_finalize(&session.encoder)
            .context("Failed to finalize loopback capture")?;

        if verbose {
            println!("🔀 Merging mic and system audio...");
        }
        merge_wav_files(&mic_wav_path, &loopback_wav_path, &wav_file_path)
            .context("Failed to merge mic and loopback audio")?;

        // Clean up temp files
        let _ = std::fs::remove_file(&mic_wav_path);
        let _ = std::fs::remove_file(&loopback_wav_path);
    } else if use_loopback {
        // Loopback was requested but failed to start — mic_temp.wav is all we have.
        // Move it to the expected recording.wav path.
        std::fs::rename(&mic_wav_path, &wav_file_path)
            .context("Failed to rename mic temp file")?;
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
        if auto_stopped {
            println!("🔇 Recording auto-stopped due to silence: {}", metadata_file_path.display());
        } else {
            println!("✅ Recording complete: {}", metadata_file_path.display());
        }
    }

    Ok(RecordingResult {
        recording_name: output_path.to_string_lossy().to_string(),
        auto_stopped,
    })
}

/// Record audio from the default input device.
pub async fn record_audio(output_path: PathBuf, options: RecordOptions) -> Result<RecordingResult> {
    let silence_flag = Arc::new(AtomicBool::new(false));
    let silence_timeout = options.silence_timeout;

    // Timestamp of last audio callback — used to detect stale device (lid close)
    let now_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64;
    let last_callback_ms = Arc::new(AtomicU64::new(now_ms));

    let wait_strategy: Box<dyn FnOnce()> = match options.stop_rx {
        Some(mut rx) => {
            let sf = silence_flag.clone();
            let lcb = last_callback_ms.clone();
            Box::new(move || {
                let (tx, rx_done) = std::sync::mpsc::channel();
                let tx2 = tx.clone();
                // Thread 1: user stop signal
                std::thread::spawn(move || {
                    let _ = rx.blocking_recv();
                    let _ = tx.send(());
                });
                // Thread 2: silence + stale callback poll
                if let Some(timeout) = silence_timeout {
                    let timeout_ms = timeout.as_millis() as u64;
                    std::thread::spawn(move || loop {
                        // Check silence flag (mic active but quiet)
                        if sf.load(Ordering::Relaxed) {
                            let _ = tx2.send(());
                            return;
                        }
                        // Check stale callback (device suspended, e.g. lid close)
                        let last = lcb.load(Ordering::Relaxed);
                        let now = SystemTime::now()
                            .duration_since(UNIX_EPOCH)
                            .unwrap_or_default()
                            .as_millis() as u64;
                        if now.saturating_sub(last) >= timeout_ms {
                            sf.store(true, Ordering::Relaxed);
                            let _ = tx2.send(());
                            return;
                        }
                        std::thread::sleep(Duration::from_millis(500));
                    });
                }
                let _ = rx_done.recv();
            })
        }
        None => {
            let handle = tokio::runtime::Handle::current();
            let sf = silence_flag.clone();
            let lcb = last_callback_ms.clone();
            Box::new(move || {
                let (tx, rx) = std::sync::mpsc::channel();
                let tx2 = tx.clone();
                let handle_clone = handle.clone();
                // Thread 1: ctrl+c
                std::thread::spawn(move || {
                    let _ = handle_clone.block_on(signal::ctrl_c());
                    let _ = tx.send(());
                });
                // Thread 2: silence + stale callback poll
                if let Some(timeout) = silence_timeout {
                    let timeout_ms = timeout.as_millis() as u64;
                    std::thread::spawn(move || loop {
                        if sf.load(Ordering::Relaxed) {
                            let _ = tx2.send(());
                            return;
                        }
                        let last = lcb.load(Ordering::Relaxed);
                        let now = SystemTime::now()
                            .duration_since(UNIX_EPOCH)
                            .unwrap_or_default()
                            .as_millis() as u64;
                        if now.saturating_sub(last) >= timeout_ms {
                            sf.store(true, Ordering::Relaxed);
                            let _ = tx2.send(());
                            return;
                        }
                        std::thread::sleep(Duration::from_millis(500));
                    });
                }
                let _ = rx.recv();
            })
        }
    };

    record_core(
        output_path,
        options.compression_settings,
        options.level_tx,
        options.verbose,
        silence_timeout,
        silence_flag,
        last_callback_ms,
        wait_strategy,
        options.input_device,
        options.loopback_device,
    )
}

// Sample handlers
fn write_input_data_f32(
    input: &[f32],
    encoder: &AudioEncoderHandle,
    level_monitor: &Arc<Mutex<AudioLevelMonitor>>,
    level_tx: Option<&mpsc::Sender<f32>>,
    silence_flag: &Arc<AtomicBool>,
    silence_timeout: Option<Duration>,
    last_callback_ms: &Arc<AtomicU64>,
) {
    // Update last callback timestamp for stale-callback detection (lid close)
    let now_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64;
    last_callback_ms.store(now_ms, Ordering::Relaxed);

    if let Ok(mut monitor) = level_monitor.try_lock() {
        if let Some(level) = monitor.update_level(input) {
            if let Some(tx) = level_tx {
                let _ = tx.try_send(level);
            }
        }
        // Check silence auto-stop
        if let Some(timeout) = silence_timeout {
            if monitor.silence_duration() >= timeout {
                silence_flag.store(true, Ordering::Relaxed);
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
    silence_flag: &Arc<AtomicBool>,
    silence_timeout: Option<Duration>,
    last_callback_ms: &Arc<AtomicU64>,
) {
    let f32_samples: Vec<f32> = input.iter().map(|&s| s as f32 / i16::MAX as f32).collect();
    write_input_data_f32(&f32_samples, encoder, level_monitor, level_tx, silence_flag, silence_timeout, last_callback_ms);
}

fn write_input_data_i32(
    input: &[i32],
    encoder: &AudioEncoderHandle,
    level_monitor: &Arc<Mutex<AudioLevelMonitor>>,
    level_tx: Option<&mpsc::Sender<f32>>,
    silence_flag: &Arc<AtomicBool>,
    silence_timeout: Option<Duration>,
    last_callback_ms: &Arc<AtomicU64>,
) {
    let f32_samples: Vec<f32> = input.iter().map(|&s| s as f32 / i32::MAX as f32).collect();
    write_input_data_f32(&f32_samples, encoder, level_monitor, level_tx, silence_flag, silence_timeout, last_callback_ms);
}

fn write_input_data_i8(
    input: &[i8],
    encoder: &AudioEncoderHandle,
    level_monitor: &Arc<Mutex<AudioLevelMonitor>>,
    level_tx: Option<&mpsc::Sender<f32>>,
    silence_flag: &Arc<AtomicBool>,
    silence_timeout: Option<Duration>,
    last_callback_ms: &Arc<AtomicU64>,
) {
    let f32_samples: Vec<f32> = input.iter().map(|&s| s as f32 / i8::MAX as f32).collect();
    write_input_data_f32(&f32_samples, encoder, level_monitor, level_tx, silence_flag, silence_timeout, last_callback_ms);
}
