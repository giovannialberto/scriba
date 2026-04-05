//! Voice-activated recording engine ("Scriba Forever" mode).
//!
//! Listens for wake phrases ("scriba record" / "scriba stop") using
//! energy-based VAD + sherpa-onnx whisper-tiny, then emits commands to the TUI.

use anyhow::Result;
use cpal::traits::{DeviceTrait, StreamTrait};
use sherpa_onnx::OfflineRecognizer;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::mpsc;

use super::audio::{create_encoder, AudioEncoder, AudioFormat, CompressionSettings};
use super::config::{LocalModel, VoiceConfig};
use super::recording::{resolve_input_device, AudioLevelMonitor};
use super::ring_buffer::RingBuffer;
use super::transcription::ensure_sherpa_model;
use crate::utils::BASE_PATH;

/// Commands emitted by the voice detector.
#[derive(Debug, Clone)]
pub enum VoiceCommand {
    /// "scriba record" detected — start recording.
    Record,
    /// "scriba stop" detected — stop recording.
    Stop,
}

/// Current operating mode of the voice detector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VoiceMode {
    /// Listening for "scriba record".
    Standby,
    /// Recording audio and listening for "scriba stop".
    Recording,
}

/// Listening state for UI display.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VoiceListeningState {
    /// Idle, waiting for speech.
    Standby,
    /// Detected speech, running whisper on a chunk.
    Processing,
    /// Voice-triggered recording is active.
    Recording,
}

/// Handle returned by `VoiceDetector::start()` to control the detector.
pub struct VoiceDetectorHandle {
    /// Set to true to request shutdown.
    shutdown: Arc<AtomicBool>,
    /// The cpal stream — kept alive.
    _stream: cpal::Stream,
    /// Current mode (shared with callback).
    mode: Arc<Mutex<VoiceMode>>,
    /// The ring buffer (shared with callback).
    ring_buffer: Arc<Mutex<RingBuffer>>,
    /// Active encoder when recording (shared with callback).
    encoder: Arc<Mutex<Option<Box<dyn AudioEncoder>>>>,
    /// Path to current recording WAV file.
    recording_wav_path: Arc<Mutex<Option<PathBuf>>>,
    /// Recording directory name.
    recording_dir_name: Arc<Mutex<Option<String>>>,
    /// Sample rate of the audio device.
    sample_rate: u32,
    /// Number of channels.
    channels: u16,
    /// Listening state for UI.
    listening_state: Arc<Mutex<VoiceListeningState>>,
    /// Volume level for UI.
    current_level: Arc<Mutex<f32>>,
}

impl VoiceDetectorHandle {
    /// Get current listening state.
    pub fn listening_state(&self) -> VoiceListeningState {
        *self.listening_state.lock().unwrap()
    }

    /// Get current audio level.
    pub fn current_level(&self) -> f32 {
        *self.current_level.lock().unwrap()
    }

    /// Drain the ring buffer contents (pre-buffer audio).
    pub fn drain_pre_buffer(&self) -> Vec<f32> {
        self.ring_buffer.lock().unwrap().drain_all()
    }

    /// Switch to recording mode: create encoder and start writing audio.
    pub fn start_recording(&self, recording_name: &str) -> Result<()> {
        let wav_dir = BASE_PATH.join(recording_name);
        std::fs::create_dir_all(&wav_dir)?;
        let wav_path = wav_dir.join("recording.wav");

        let settings = CompressionSettings {
            format: AudioFormat::Wav,
            sample_rate: self.sample_rate,
            bitrate_kbps: None,
            channels: self.channels,
            speech_optimized: false,
        };

        let encoder = create_encoder(&wav_path, &settings)?;

        // Write pre-buffer contents to encoder
        let pre_buf = self.ring_buffer.lock().unwrap().drain_all();
        {
            let mut enc_guard = self.encoder.lock().unwrap();
            *enc_guard = Some(encoder);
            if let Some(ref mut enc) = *enc_guard {
                if !pre_buf.is_empty() {
                    let _ = enc.encode_samples(&pre_buf);
                }
            }
        }

        *self.recording_wav_path.lock().unwrap() = Some(wav_path);
        *self.recording_dir_name.lock().unwrap() = Some(recording_name.to_string());
        *self.mode.lock().unwrap() = VoiceMode::Recording;
        *self.listening_state.lock().unwrap() = VoiceListeningState::Recording;

        Ok(())
    }

    /// Stop recording and finalize the WAV file.
    /// Returns (directory_name, wav_path) for the pipeline to process.
    pub fn stop_recording(&self) -> Result<Option<(String, PathBuf)>> {
        *self.mode.lock().unwrap() = VoiceMode::Standby;
        *self.listening_state.lock().unwrap() = VoiceListeningState::Standby;

        let mut enc_guard = self.encoder.lock().unwrap();
        if let Some(mut enc) = enc_guard.take() {
            enc.finalize()?;
        }

        let wav_path = self.recording_wav_path.lock().unwrap().take();
        let dir_name = self.recording_dir_name.lock().unwrap().take();

        match (dir_name, wav_path) {
            (Some(dir), Some(path)) => Ok(Some((dir, path))),
            _ => Ok(None),
        }
    }

    /// Shut down the voice detector.
    pub fn shutdown(&self) {
        self.shutdown.store(true, Ordering::Relaxed);
    }
}

/// Start the voice detection engine.
///
/// Opens the default audio input device, listens for speech using VAD,
/// and runs whisper-tiny on detected speech chunks to identify wake phrases.
/// Sends `VoiceCommand` messages through the provided channel.
///
/// Note: Voice mode operates on a single mic stream for VAD/wake-word detection.
/// System audio loopback capture is used only during standard recordings (see
/// `recording.rs` and `loopback.rs`), not during voice detection.
pub async fn start_voice_detector(
    config: &VoiceConfig,
    command_tx: mpsc::Sender<VoiceCommand>,
    input_device_name: Option<&str>,
) -> Result<VoiceDetectorHandle> {
    // Ensure whisper-tiny model is available for wake-word detection
    let model_paths = ensure_sherpa_model(LocalModel::WhisperTiny, true).await?;

    let device = resolve_input_device(input_device_name)?;
    let device_config = device
        .default_input_config()
        .map_err(|e| anyhow::anyhow!("Failed to get default input config: {}", e))?;

    let sample_rate = device_config.sample_rate().0;
    let channels = device_config.channels();
    let pre_buffer_capacity = (sample_rate as f32 * config.pre_buffer_seconds) as usize;

    let shutdown = Arc::new(AtomicBool::new(false));
    let mode = Arc::new(Mutex::new(VoiceMode::Standby));
    let ring_buffer = Arc::new(Mutex::new(RingBuffer::new(pre_buffer_capacity)));
    let encoder: Arc<Mutex<Option<Box<dyn AudioEncoder>>>> = Arc::new(Mutex::new(None));
    let listening_state = Arc::new(Mutex::new(VoiceListeningState::Standby));
    let current_level = Arc::new(Mutex::new(0.0f32));
    let recording_wav_path: Arc<Mutex<Option<PathBuf>>> = Arc::new(Mutex::new(None));
    let recording_dir_name: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));

    // Speech accumulator state (shared with callback)
    let speech_buffer: Arc<Mutex<Vec<f32>>> = Arc::new(Mutex::new(Vec::new()));
    let speech_active = Arc::new(AtomicBool::new(false));
    let silence_start_ms = Arc::new(Mutex::new(None::<u64>));
    let vad_threshold = config.vad_threshold;
    let silence_after_speech_ms: u64 = 800;

    // Channel to send speech chunks to the processing thread
    let (chunk_tx, mut chunk_rx) = mpsc::channel::<(Vec<f32>, VoiceMode)>(8);

    // Clones for the audio callback
    let cb_ring_buffer = ring_buffer.clone();
    let cb_encoder = encoder.clone();
    let cb_mode = mode.clone();
    let cb_speech_buffer = speech_buffer.clone();
    let cb_speech_active = speech_active.clone();
    let cb_silence_start = silence_start_ms.clone();
    let cb_listening_state = listening_state.clone();
    let cb_current_level = current_level.clone();
    let cb_shutdown = shutdown.clone();
    let cb_chunk_tx = chunk_tx.clone();

    let level_monitor = Arc::new(Mutex::new(AudioLevelMonitor::new()));

    let err_fn = |err| {
        eprintln!("Voice detector stream error: {}", err);
    };

    // Build the audio callback. We handle all sample formats by converting to f32.
    let build_callback = move |data: &[f32]| {
        if cb_shutdown.load(Ordering::Relaxed) {
            return;
        }

        // Always feed ring buffer
        if let Ok(mut rb) = cb_ring_buffer.try_lock() {
            rb.push_samples(data);
        }

        // If recording, also write to encoder
        let current_mode = *cb_mode.lock().unwrap();
        if current_mode == VoiceMode::Recording {
            if let Ok(mut enc_guard) = cb_encoder.try_lock() {
                if let Some(ref mut enc) = *enc_guard {
                    let _ = enc.encode_samples(data);
                }
            }
        }

        // Update level monitor for UI feedback
        if let Ok(mut monitor) = level_monitor.try_lock() {
            if let Some(level) = monitor.update_level(data) {
                if let Ok(mut lvl) = cb_current_level.try_lock() {
                    *lvl = level;
                }
            }
        }

        // VAD: compute RMS
        let sum_sq: f32 = data.iter().map(|&s| s * s).sum();
        let rms = (sum_sq / data.len() as f32).sqrt();

        let now_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        if rms >= vad_threshold {
            // Speech detected
            cb_speech_active.store(true, Ordering::Relaxed);
            *cb_silence_start.lock().unwrap() = None;

            // Accumulate speech audio
            if let Ok(mut buf) = cb_speech_buffer.try_lock() {
                buf.extend_from_slice(data);
                // Safety cap: don't accumulate more than 15 seconds
                let max_samples = sample_rate as usize * 15;
                if buf.len() > max_samples {
                    let drain_end = buf.len() - max_samples;
                    buf.drain(..drain_end);
                }
            }
        } else if cb_speech_active.load(Ordering::Relaxed) {
            // Below threshold but was in speech — track silence duration
            let mut silence_guard = cb_silence_start.lock().unwrap();
            let silence_start = silence_guard.get_or_insert(now_ms);

            // Also keep accumulating during the silence gap
            if let Ok(mut buf) = cb_speech_buffer.try_lock() {
                buf.extend_from_slice(data);
            }

            if now_ms.saturating_sub(*silence_start) >= silence_after_speech_ms {
                // Speech chunk complete — send to processing thread
                cb_speech_active.store(false, Ordering::Relaxed);
                *silence_guard = None;

                if let Ok(mut buf) = cb_speech_buffer.try_lock() {
                    if !buf.is_empty() {
                        let chunk: Vec<f32> = buf.drain(..).collect();
                        // Update listening state
                        if let Ok(mut ls) = cb_listening_state.try_lock() {
                            if *ls != VoiceListeningState::Recording {
                                *ls = VoiceListeningState::Processing;
                            }
                        }
                        let _ = cb_chunk_tx.try_send((chunk, current_mode));
                    }
                }
            }
        }
    };

    // Build stream based on sample format
    let stream = match device_config.sample_format() {
        cpal::SampleFormat::F32 => {
            device.build_input_stream(
                &device_config.clone().into(),
                move |data: &[f32], _: &_| build_callback(data),
                err_fn,
                None,
            )?
        }
        cpal::SampleFormat::I16 => {
            device.build_input_stream(
                &device_config.clone().into(),
                move |data: &[i16], _: &_| {
                    let f32_data: Vec<f32> = data.iter().map(|&s| s as f32 / i16::MAX as f32).collect();
                    build_callback(&f32_data);
                },
                err_fn,
                None,
            )?
        }
        cpal::SampleFormat::I32 => {
            device.build_input_stream(
                &device_config.clone().into(),
                move |data: &[i32], _: &_| {
                    let f32_data: Vec<f32> = data.iter().map(|&s| s as f32 / i32::MAX as f32).collect();
                    build_callback(&f32_data);
                },
                err_fn,
                None,
            )?
        }
        cpal::SampleFormat::I8 => {
            device.build_input_stream(
                &device_config.clone().into(),
                move |data: &[i8], _: &_| {
                    let f32_data: Vec<f32> = data.iter().map(|&s| s as f32 / i8::MAX as f32).collect();
                    build_callback(&f32_data);
                },
                err_fn,
                None,
            )?
        }
        fmt => {
            return Err(anyhow::anyhow!(
                "Unsupported sample format for voice detection: {:?}",
                fmt
            ));
        }
    };

    stream.play()?;

    // Spawn a dedicated std::thread that owns the recognizer (not Send/Sync).
    // This avoids recreating the recognizer for every voice chunk.
    let model_dir = model_paths.dir.clone();
    let (work_tx, work_rx) = std::sync::mpsc::channel::<(Vec<f32>, u32, tokio::sync::oneshot::Sender<Result<String>>)>();

    std::thread::Builder::new()
        .name("voice-recognizer".into())
        .spawn(move || {
            let config = super::transcription::build_recognizer_config(
                LocalModel::WhisperTiny,
                &model_dir,
            );
            let recognizer = match OfflineRecognizer::create(&config) {
                Some(r) => r,
                None => {
                    // Drain requests with error if recognizer creation fails
                    while let Ok((_, _, reply)) = work_rx.recv() {
                        let _ = reply.send(Err(anyhow::anyhow!("Failed to create recognizer")));
                    }
                    return;
                }
            };
            while let Ok((chunk, sr, reply)) = work_rx.recv() {
                let result = run_sherpa_on_chunk(&recognizer, &chunk, sr);
                let _ = reply.send(result);
            }
        })?;

    // Async task: receives speech chunks from audio callback, dispatches to recognizer thread
    let proc_listening_state = listening_state.clone();
    let proc_shutdown = shutdown.clone();

    tokio::spawn(async move {
        while let Some((chunk, mode_at_capture)) = chunk_rx.recv().await {
            if proc_shutdown.load(Ordering::Relaxed) {
                break;
            }

            let (reply_tx, reply_rx) = tokio::sync::oneshot::channel();
            if work_tx.send((chunk, sample_rate, reply_tx)).is_err() {
                break; // recognizer thread exited
            }

            let transcript = reply_rx.await;

            // Reset listening state back from Processing
            if let Ok(mut ls) = proc_listening_state.try_lock() {
                if *ls == VoiceListeningState::Processing {
                    *ls = VoiceListeningState::Standby;
                }
            }

            if let Ok(Ok(text)) = transcript {
                let text_lower = text.to_lowercase();
                match mode_at_capture {
                    VoiceMode::Standby => {
                        if text_lower.contains("scriba record")
                            || text_lower.contains("scriba, record")
                            || text_lower.contains("scriba. record")
                        {
                            let _ = command_tx.send(VoiceCommand::Record).await;
                        }
                    }
                    VoiceMode::Recording => {
                        if text_lower.contains("scriba stop")
                            || text_lower.contains("scriba, stop")
                            || text_lower.contains("scriba. stop")
                        {
                            let _ = command_tx.send(VoiceCommand::Stop).await;
                        }
                    }
                }
            }
        }
    });

    Ok(VoiceDetectorHandle {
        shutdown,
        _stream: stream,
        mode,
        ring_buffer,
        encoder,
        recording_wav_path,
        recording_dir_name,
        sample_rate,
        channels,
        listening_state,
        current_level,
    })
}

/// Run sherpa-onnx inference on a raw f32 audio chunk.
///
/// The chunk is at the device's native sample rate; sherpa-onnx needs 16kHz mono.
/// We do a simple linear resampling here.
fn run_sherpa_on_chunk(recognizer: &OfflineRecognizer, samples: &[f32], sample_rate: u32) -> Result<String> {
    // Resample to 16kHz if needed
    let samples_16k = if sample_rate != 16000 {
        resample_to_16k(samples, sample_rate)
    } else {
        samples.to_vec()
    };

    let stream = recognizer.create_stream();
    stream.accept_waveform(16000, &samples_16k);
    recognizer.decode(&stream);

    let result = stream.get_result()
        .ok_or_else(|| anyhow::anyhow!("Failed to get recognition result for voice chunk"))?;

    Ok(result.text.trim().to_string())
}

/// Simple linear resampling from source sample rate to 16kHz.
fn resample_to_16k(samples: &[f32], source_rate: u32) -> Vec<f32> {
    let ratio = 16000.0 / source_rate as f64;
    let output_len = (samples.len() as f64 * ratio) as usize;
    let mut output = Vec::with_capacity(output_len);

    for i in 0..output_len {
        let src_idx = i as f64 / ratio;
        let idx = src_idx as usize;
        let frac = src_idx - idx as f64;

        let sample = if idx + 1 < samples.len() {
            samples[idx] as f64 * (1.0 - frac) + samples[idx + 1] as f64 * frac
        } else if idx < samples.len() {
            samples[idx] as f64
        } else {
            0.0
        };

        output.push(sample as f32);
    }

    output
}
