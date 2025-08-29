use crate::audio::{
    convert_wav_to_mp3, create_encoder, AudioEncoder, AudioFormat, CompressionSettings,
};
use crate::database::{Database, Recording};
use anyhow::Context;
use chrono::Utc;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use dirs::home_dir;
use lazy_static::lazy_static;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::signal;
use tokio::sync::mpsc;

lazy_static! {
    static ref BASE_PATH: PathBuf = {
        let path = home_dir()
            .expect("error home dir")
            .join("scriba_recordings");
        if !path.exists() {
            std::fs::create_dir_all(&path).expect("Failed to create directory");
        }
        path
    };
}

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

    pub fn update_level(&mut self, samples: &[f32]) -> Option<f32> {
        // Calculate RMS (Root Mean Square) for audio level
        let sum_of_squares: f32 = samples.iter().map(|&s| s * s).sum();
        let rms = (sum_of_squares / samples.len() as f32).sqrt();

        // Smooth the level with a simple moving average
        self.level = self.level * 0.8 + rms * 0.2;

        // Return level every 100ms for TUI updates
        if self.last_update.elapsed() >= Duration::from_millis(100) {
            self.last_update = Instant::now();
            Some(self.level)
        } else {
            None
        }
    }

    // Legacy terminal VU output removed; TUI shows levels now
}

// Main recording function
pub async fn record(
    output_path: PathBuf,
    compression_settings: Option<CompressionSettings>,
) -> Result<(), anyhow::Error> {
    // Get the default input device
    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .expect("No input device available");

    // Get the device default configuration
    let config = device
        .default_input_config()
        .expect("Failed to get default input config");
    println!("Default input config: {:?}", config);

    // Save config values for database insertion (before they're consumed)
    let (sample_rate, channels) = if let Some(ref settings) = compression_settings {
        (settings.sample_rate as i64, settings.channels as i64)
    } else {
        (config.sample_rate().0 as i64, config.channels() as i64)
    };

    // Always record as WAV first (for perfect quality), then convert if needed
    let wav_file_path = BASE_PATH.join(&output_path).join("recording.wav");

    // Determine final file path and format
    let (final_file_path, audio_format_str, needs_conversion) =
        if let Some(ref settings) = compression_settings {
            match settings.format {
                AudioFormat::Mp3 => {
                    let mp3_path = BASE_PATH.join(&output_path).join("recording.mp3");
                    (mp3_path, "mp3".to_string(), true)
                }
                _ => {
                    let filename = settings.get_filename("recording");
                    (
                        BASE_PATH.join(&output_path).join(filename),
                        settings.format.to_string().to_lowercase(),
                        false,
                    )
                }
            }
        } else {
            (wav_file_path.clone(), "wav".to_string(), false)
        };

    // Ensure the recording directory exists
    if let Some(parent) = wav_file_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    // Always create WAV encoder with device settings (perfect quality recording)
    let recording_settings = CompressionSettings {
        format: AudioFormat::Wav,
        sample_rate: config.sample_rate().0,
        bitrate_kbps: None,
        channels: config.channels(),
        speech_optimized: false,
    };
    let encoder = create_encoder(&wav_file_path, &recording_settings)?;
    let encoder = Arc::new(Mutex::new(Some(encoder)));

    // Create audio level monitor
    let level_monitor = Arc::new(Mutex::new(AudioLevelMonitor::new()));

    // A flag to indicate that recording is in progress.
    println!("Begin recording... (Press Ctrl+C to stop)");
    println!();

    // Run the input stream on a separate thread.
    let encoder_2 = encoder.clone();
    let level_monitor_2 = level_monitor.clone();

    let err_fn = move |err| {
        eprintln!("an error occurred on stream: {}", err);
    };

    let stream = match config.sample_format() {
        cpal::SampleFormat::I8 => device.build_input_stream(
            &config.into(),
            move |data, _: &_| {
                write_input_data_with_monitoring_i8(data, &encoder_2, &level_monitor_2)
            },
            err_fn,
            None,
        )?,
        cpal::SampleFormat::I16 => device.build_input_stream(
            &config.into(),
            move |data, _: &_| {
                write_input_data_with_monitoring_i16(data, &encoder_2, &level_monitor_2)
            },
            err_fn,
            None,
        )?,
        cpal::SampleFormat::I32 => device.build_input_stream(
            &config.into(),
            move |data, _: &_| {
                write_input_data_with_monitoring_i32(data, &encoder_2, &level_monitor_2)
            },
            err_fn,
            None,
        )?,
        cpal::SampleFormat::F32 => device.build_input_stream(
            &config.into(),
            move |data, _: &_| {
                write_input_data_with_monitoring_f32(data, &encoder_2, &level_monitor_2)
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

    signal::ctrl_c().await.context("ctrl c error")?;

    drop(stream);

    // Clear the level display line and show completion message
    print!("\r");
    println!("Recording {} complete!", wav_file_path.display());

    // Finalize the encoder
    if let Some(encoder) = encoder.lock().unwrap().as_mut() {
        encoder.finalize()?;
    }

    // Post-recording conversion if needed
    if needs_conversion {
        if let Some(ref settings) = compression_settings {
            convert_wav_to_mp3(&wav_file_path, &final_file_path, settings)
                .context("Failed to convert WAV to MP3")?;

            // Optionally remove the WAV file after successful conversion
            std::fs::remove_file(&wav_file_path).context("Failed to remove temporary WAV file")?;
        }
    }

    // Use the final file path for metadata
    let metadata_file_path = if needs_conversion {
        &final_file_path
    } else {
        &wav_file_path
    };

    // Save recording metadata to database
    let mut db = Database::new().context("Failed to connect to database")?;

    // Get file size and duration info
    let file_metadata = std::fs::metadata(metadata_file_path)?;
    let file_size_bytes = file_metadata.len() as i64;

    // Calculate duration based on file format
    let duration_seconds = calculate_audio_duration(metadata_file_path, sample_rate, channels)?;

    // Create recording record
    let recording = Recording {
        id: None,
        directory_name: output_path.to_string_lossy().to_string(),
        display_name: None,
        created_at: Utc::now(),
        updated_at: Utc::now(),
        duration_seconds: Some(duration_seconds),
        file_size_bytes: Some(file_size_bytes),
        audio_format: audio_format_str,
        sample_rate,
        channels,
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
        audio_path: metadata_file_path
            .file_name()
            .unwrap()
            .to_string_lossy()
            .to_string(),
        transcript_path: None,
    };

    match db.insert_recording(&recording) {
        Ok(id) => println!("📊 Recording saved to database with ID: {}", id),
        Err(e) => eprintln!("⚠️ Warning: Failed to save recording to database: {}", e),
    }

    Ok(())
}

type AudioEncoderHandle = Arc<Mutex<Option<Box<dyn AudioEncoder>>>>;

// Specialized functions for level monitoring with audio level feedback
fn write_input_data_with_monitoring_f32(
    input: &[f32],
    encoder: &AudioEncoderHandle,
    level_monitor: &Arc<Mutex<AudioLevelMonitor>>,
) {
    // Update the level monitor with f32 samples
    if let Ok(mut monitor) = level_monitor.try_lock() {
        monitor.update_level(input);
    }

    // Write audio data
    if let Ok(mut guard) = encoder.try_lock() {
        if let Some(encoder) = guard.as_mut() {
            encoder.encode_samples(input).ok();
        }
    }
}

fn write_input_data_with_monitoring_i16(
    input: &[i16],
    encoder: &AudioEncoderHandle,
    level_monitor: &Arc<Mutex<AudioLevelMonitor>>,
) {
    // Convert i16 to f32 for level calculation and encoding
    let f32_samples: Vec<f32> = input.iter().map(|&s| s as f32 / i16::MAX as f32).collect();

    if let Ok(mut monitor) = level_monitor.try_lock() {
        monitor.update_level(&f32_samples);
    }

    // Write audio data
    if let Ok(mut guard) = encoder.try_lock() {
        if let Some(encoder) = guard.as_mut() {
            encoder.encode_samples(&f32_samples).ok();
        }
    }
}

fn write_input_data_with_monitoring_i32(
    input: &[i32],
    encoder: &AudioEncoderHandle,
    level_monitor: &Arc<Mutex<AudioLevelMonitor>>,
) {
    // Convert i32 to f32 for level calculation and encoding
    let f32_samples: Vec<f32> = input.iter().map(|&s| s as f32 / i32::MAX as f32).collect();

    if let Ok(mut monitor) = level_monitor.try_lock() {
        monitor.update_level(&f32_samples);
    }

    // Write audio data
    if let Ok(mut guard) = encoder.try_lock() {
        if let Some(encoder) = guard.as_mut() {
            encoder.encode_samples(&f32_samples).ok();
        }
    }
}

fn write_input_data_with_monitoring_i8(
    input: &[i8],
    encoder: &AudioEncoderHandle,
    level_monitor: &Arc<Mutex<AudioLevelMonitor>>,
) {
    // Convert i8 to f32 for level calculation and encoding
    let f32_samples: Vec<f32> = input.iter().map(|&s| s as f32 / i8::MAX as f32).collect();

    if let Ok(mut monitor) = level_monitor.try_lock() {
        monitor.update_level(&f32_samples);
    }

    // Write audio data
    if let Ok(mut guard) = encoder.try_lock() {
        if let Some(encoder) = guard.as_mut() {
            encoder.encode_samples(&f32_samples).ok();
        }
    }
}

pub fn calculate_audio_duration(
    file_path: &std::path::Path,
    _sample_rate: i64,
    _channels: i64,
) -> Result<i64, anyhow::Error> {
    let extension = file_path
        .extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or("");

    match extension.to_lowercase().as_str() {
        "wav" => {
            // Use hound to properly read the WAV file header and get accurate info
            let reader = hound::WavReader::open(file_path)
                .context("Failed to open WAV file for duration calculation")?;

            let spec = reader.spec();
            let wav_sample_rate = spec.sample_rate as i64;

            // Get the total number of samples
            let duration_samples = reader.duration() as i64;

            // Calculate duration in seconds
            let duration_seconds = duration_samples / wav_sample_rate;

            Ok(duration_seconds)
        }
        "mp3" => {
            // For MP3, use ffprobe for accurate duration calculation
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
                    // Fallback: estimate using 32kbps bitrate (our default)
                    let file_size = std::fs::metadata(file_path)?.len() as i64;
                    // 32kbps = 32000 bits per second = 4000 bytes per second
                    let estimated_duration = file_size / 4000;
                    Ok(estimated_duration.max(1))
                }
            }
        }
        _ => {
            // Fallback: estimate based on provided sample rate and assume reasonable file size
            Ok(1) // Default to 1 second for unknown formats
        }
    }
}

// Controlled recording function for TUI usage
pub async fn record_with_control(
    output_path: PathBuf,
    compression_settings: Option<CompressionSettings>,
    mut stop_rx: mpsc::Receiver<()>,
    level_tx: mpsc::Sender<f32>,
) -> Result<String, anyhow::Error> {
    // Get the default input device
    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .ok_or_else(|| anyhow::anyhow!("No input device available"))?;

    // Get the device default configuration
    let config = device
        .default_input_config()
        .map_err(|e| anyhow::anyhow!("Failed to get default input config: {}", e))?;

    // Save config values for database insertion (before they're consumed)
    let (sample_rate, channels) = if let Some(ref settings) = compression_settings {
        (settings.sample_rate as i64, settings.channels as i64)
    } else {
        (config.sample_rate().0 as i64, config.channels() as i64)
    };

    // Always record as WAV first (for perfect quality), then convert if needed
    let wav_file_path = BASE_PATH.join(&output_path).join("recording.wav");

    // Determine final file path and format
    let (final_file_path, audio_format_str, needs_conversion) =
        if let Some(ref settings) = compression_settings {
            match settings.format {
                AudioFormat::Mp3 => {
                    let mp3_path = BASE_PATH.join(&output_path).join("recording.mp3");
                    (mp3_path, "mp3".to_string(), true)
                }
                _ => {
                    let filename = settings.get_filename("recording");
                    (
                        BASE_PATH.join(&output_path).join(filename),
                        settings.format.to_string().to_lowercase(),
                        false,
                    )
                }
            }
        } else {
            (wav_file_path.clone(), "wav".to_string(), false)
        };

    // Ensure the recording directory exists
    if let Some(parent) = wav_file_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    // Always create WAV encoder with device settings (perfect quality recording)
    let recording_settings = CompressionSettings {
        format: AudioFormat::Wav,
        sample_rate: config.sample_rate().0,
        bitrate_kbps: None,
        channels: config.channels(),
        speech_optimized: false,
    };
    let encoder = create_encoder(&wav_file_path, &recording_settings)?;
    let encoder = Arc::new(Mutex::new(Some(encoder)));

    // Create audio level monitor for TUI feedback
    let level_monitor = Arc::new(Mutex::new(AudioLevelMonitor::new()));

    // Set up the stream
    let encoder_2 = encoder.clone();
    let level_monitor_2 = level_monitor.clone();
    let level_tx_2 = level_tx.clone();

    let err_fn = move |_err| {
        // Silently handle stream errors in TUI mode
    };

    let stream = match config.sample_format() {
        cpal::SampleFormat::F32 => device.build_input_stream(
            &config.into(),
            move |data, _: &_| {
                write_input_data_with_level_feedback_f32(
                    data,
                    &encoder_2,
                    &level_monitor_2,
                    &level_tx_2,
                )
            },
            err_fn,
            None,
        )?,
        // Add other sample formats as needed
        sample_format => {
            return Err(anyhow::Error::msg(format!(
                "Unsupported sample format '{sample_format}' for TUI recording"
            )))
        }
    };

    stream.play()?;

    // Wait for stop signal in a blocking way to avoid Send issues
    let stop_received = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let _stop_received_clone = stop_received.clone();

    // Use a std thread to receive the stop signal
    let (stop_tx, stop_rx_std) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let _ = stop_rx.recv().await;
            let _ = stop_tx.send(());
        });
    });

    // Wait for stop signal
    let _ = stop_rx_std.recv();
    stop_received.store(true, std::sync::atomic::Ordering::Relaxed);

    // Stop the stream
    drop(stream);

    // Finalize the encoder
    if let Some(mut encoder) = encoder.lock().unwrap().take() {
        encoder.finalize()?;
    }

    // Convert to compressed format if needed
    if needs_conversion {
        if let Some(settings) = compression_settings {
            convert_wav_to_mp3(&wav_file_path, &final_file_path, &settings)?;
            // Remove the intermediate WAV file
            let _ = std::fs::remove_file(&wav_file_path);
        }
    }

    // Store recording metadata in database
    let recording_duration = calculate_audio_duration(&final_file_path, sample_rate, channels)?;

    let recording = Recording {
        id: None,
        directory_name: output_path.to_string_lossy().to_string(),
        display_name: None,
        created_at: Utc::now(),
        updated_at: Utc::now(),
        duration_seconds: Some(recording_duration),
        file_size_bytes: Some(std::fs::metadata(&final_file_path)?.len() as i64),
        audio_format: audio_format_str,
        sample_rate,
        channels,
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
    };

    let mut db = Database::new()?;
    let _ = db.insert_recording(&recording); // Silently handle database errors

    Ok(output_path.to_string_lossy().to_string())
}

// Audio data handler with level feedback for TUI
fn write_input_data_with_level_feedback_f32(
    input: &[f32],
    encoder: &Arc<Mutex<Option<Box<dyn AudioEncoder>>>>,
    level_monitor: &Arc<Mutex<AudioLevelMonitor>>,
    level_tx: &mpsc::Sender<f32>,
) {
    // Send to encoder
    if let Ok(mut enc_lock) = encoder.lock() {
        if let Some(ref mut enc) = *enc_lock {
            let _ = enc.encode_samples(input);
        }
    }

    // Update level and send to TUI if needed
    if let Ok(mut monitor) = level_monitor.lock() {
        if let Some(level) = monitor.update_level(input) {
            let _ = level_tx.try_send(level);
        }
    }
}
