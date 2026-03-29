//! Audio format handling and encoding for Scriba.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Supported audio formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AudioFormat {
    Wav,
    WavCompressed,
    Mp3,
}

impl AudioFormat {
    pub fn extension(&self) -> &'static str {
        match self {
            AudioFormat::Wav => "wav",
            AudioFormat::WavCompressed => "wav",
            AudioFormat::Mp3 => "mp3",
        }
    }

    pub fn mime_type(&self) -> &'static str {
        match self {
            AudioFormat::Wav => "audio/wav",
            AudioFormat::WavCompressed => "audio/wav",
            AudioFormat::Mp3 => "audio/mpeg",
        }
    }
}

impl std::fmt::Display for AudioFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            AudioFormat::Wav => "WAV",
            AudioFormat::WavCompressed => "WAV (Compressed)",
            AudioFormat::Mp3 => "MP3",
        };
        write!(f, "{}", s)
    }
}

impl std::str::FromStr for AudioFormat {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "wav" => Ok(AudioFormat::Wav),
            "wav-compressed" | "compressed" => Ok(AudioFormat::WavCompressed),
            "mp3" => Ok(AudioFormat::Mp3),
            _ => Err(anyhow::anyhow!("Unsupported audio format: {}", s)),
        }
    }
}

/// Audio compression settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionSettings {
    pub format: AudioFormat,
    pub sample_rate: u32,
    pub bitrate_kbps: Option<u32>,
    pub channels: u16,
    pub speech_optimized: bool,
}

impl Default for CompressionSettings {
    fn default() -> Self {
        Self {
            format: AudioFormat::Wav,
            sample_rate: 48000,
            bitrate_kbps: None,
            channels: 1,
            speech_optimized: false,
        }
    }
}

impl CompressionSettings {
    /// Speech-optimized preset using MP3 compression.
    /// Reduces file size by ~85-90%.
    pub fn speech_optimized() -> Self {
        Self {
            format: AudioFormat::Mp3,
            sample_rate: 22050,
            bitrate_kbps: Some(32),
            channels: 1,
            speech_optimized: true,
        }
    }

    /// Create optimized settings based on device capabilities.
    pub fn optimized_for_device(device_sample_rate: u32, _device_channels: u16) -> Self {
        let optimized_rate = match device_sample_rate {
            48000 => 24000,
            44100 => 22050,
            rate if rate >= 32000 => rate / 2,
            rate => rate,
        };

        Self {
            format: AudioFormat::Mp3,
            sample_rate: optimized_rate,
            bitrate_kbps: Some(32),
            channels: 1,
            speech_optimized: true,
        }
    }

    /// High quality preset (full quality WAV).
    pub fn high_quality() -> Self {
        Self {
            format: AudioFormat::Wav,
            sample_rate: 44100,
            bitrate_kbps: None,
            channels: 2,
            speech_optimized: false,
        }
    }

    /// Get expected file size reduction compared to full-quality WAV.
    pub fn estimated_size_reduction(&self) -> f32 {
        match self.format {
            AudioFormat::Wav => 1.0,
            AudioFormat::WavCompressed => {
                let sample_rate_reduction = 22050.0 / 48000.0;
                let channel_reduction = if self.channels == 1 { 0.5 } else { 1.0 };
                sample_rate_reduction * channel_reduction
            }
            AudioFormat::Mp3 => 0.12,
        }
    }

    /// Get filename with appropriate extension.
    pub fn get_filename(&self, base_name: &str) -> String {
        format!("{}.{}", base_name, self.format.extension())
    }
}

/// Audio encoder trait for different formats.
pub trait AudioEncoder: Send {
    fn encode_samples(&mut self, samples: &[f32]) -> Result<()>;
    fn finalize(&mut self) -> Result<()>;
}

/// Create appropriate encoder for the given settings.
pub fn create_encoder(
    output_path: &Path,
    settings: &CompressionSettings,
) -> Result<Box<dyn AudioEncoder>> {
    Ok(Box::new(WavEncoder::new(output_path, settings)?))
}

/// WAV encoder using hound crate.
pub struct WavEncoder {
    writer: hound::WavWriter<std::io::BufWriter<std::fs::File>>,
}

impl WavEncoder {
    pub fn new(output_path: &Path, settings: &CompressionSettings) -> Result<Self> {
        let spec = hound::WavSpec {
            channels: settings.channels,
            sample_rate: settings.sample_rate,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        };

        let writer =
            hound::WavWriter::create(output_path, spec).context("Failed to create WAV writer")?;

        Ok(Self { writer })
    }
}

impl AudioEncoder for WavEncoder {
    fn encode_samples(&mut self, samples: &[f32]) -> Result<()> {
        for &sample in samples {
            self.writer
                .write_sample(sample)
                .context("Failed to write WAV sample")?;
        }
        Ok(())
    }

    fn finalize(&mut self) -> Result<()> {
        Ok(())
    }
}

/// Post-recording conversion from WAV to MP3 using ffmpeg.
pub fn convert_wav_to_mp3(
    wav_path: &Path,
    mp3_path: &Path,
    settings: &CompressionSettings,
) -> Result<()> {
    let bitrate = settings.bitrate_kbps.unwrap_or(32);
    let sample_rate = settings.sample_rate;

    let output = std::process::Command::new("ffmpeg")
        .arg("-i")
        .arg(wav_path)
        .arg("-codec:a")
        .arg("libmp3lame")
        .arg("-b:a")
        .arg(format!("{}k", bitrate))
        .arg("-ar")
        .arg(sample_rate.to_string())
        .arg("-ac")
        .arg(settings.channels.to_string())
        .arg("-y")
        .arg(mp3_path)
        .output()
        .context("Failed to run ffmpeg - make sure it's installed")?;

    if !output.status.success() {
        let error_msg = String::from_utf8_lossy(&output.stderr);
        return Err(anyhow::anyhow!("FFmpeg conversion failed: {}", error_msg));
    }

    Ok(())
}

/// Merge a microphone WAV and a loopback WAV into a single mono WAV.
///
/// Handles sample rate mismatches (resamples loopback to match mic via rubato),
/// channel count mismatches (downmixes stereo to mono), and different lengths
/// (pads the shorter file with silence). No external tools needed.
pub fn merge_wav_files(
    mic_wav: &Path,
    loopback_wav: &Path,
    output_wav: &Path,
) -> Result<()> {
    let mut mic_reader =
        hound::WavReader::open(mic_wav).context("Failed to open mic WAV for merge")?;
    let mut lb_reader =
        hound::WavReader::open(loopback_wav).context("Failed to open loopback WAV for merge")?;

    let mic_spec = mic_reader.spec();
    let lb_spec = lb_reader.spec();

    let mic_rate = mic_spec.sample_rate;
    let lb_rate = lb_spec.sample_rate;

    // Read all samples as f32
    let mic_samples: Vec<f32> = read_samples_as_f32(&mut mic_reader)?;
    let lb_samples_raw: Vec<f32> = read_samples_as_f32(&mut lb_reader)?;

    // Downmix loopback to mono if stereo
    let lb_mono = if lb_spec.channels > 1 {
        downmix_to_mono(&lb_samples_raw, lb_spec.channels)
    } else {
        lb_samples_raw
    };

    // Downmix mic to mono if stereo
    let mic_mono = if mic_spec.channels > 1 {
        downmix_to_mono(&mic_samples, mic_spec.channels)
    } else {
        mic_samples
    };

    // Resample loopback to match mic sample rate if different
    let lb_resampled = if lb_rate != mic_rate {
        resample_mono(&lb_mono, lb_rate, mic_rate)?
    } else {
        lb_mono
    };

    // Mix: add samples with clamping, pad shorter with silence
    let max_len = mic_mono.len().max(lb_resampled.len());
    let mut mixed = Vec::with_capacity(max_len);
    for i in 0..max_len {
        let s1 = if i < mic_mono.len() { mic_mono[i] } else { 0.0 };
        let s2 = if i < lb_resampled.len() {
            lb_resampled[i]
        } else {
            0.0
        };
        mixed.push((s1 + s2).clamp(-1.0, 1.0));
    }

    // Write merged mono WAV
    let out_spec = hound::WavSpec {
        channels: 1,
        sample_rate: mic_rate,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut writer =
        hound::WavWriter::create(output_wav, out_spec).context("Failed to create merged WAV")?;
    for sample in &mixed {
        writer
            .write_sample(*sample)
            .context("Failed to write merged sample")?;
    }
    writer.finalize().context("Failed to finalize merged WAV")?;

    Ok(())
}

/// Read all samples from a WAV reader as f32, regardless of the source format.
fn read_samples_as_f32(reader: &mut hound::WavReader<std::io::BufReader<std::fs::File>>) -> Result<Vec<f32>> {
    let spec = reader.spec();
    match spec.sample_format {
        hound::SampleFormat::Float => {
            let samples: Result<Vec<f32>, _> = reader.samples::<f32>().collect();
            Ok(samples.context("Failed to read f32 samples")?)
        }
        hound::SampleFormat::Int => {
            match spec.bits_per_sample {
                16 => {
                    let samples: Result<Vec<f32>, _> = reader
                        .samples::<i16>()
                        .map(|s| s.map(|v| v as f32 / 32768.0))
                        .collect();
                    Ok(samples.context("Failed to read i16 samples")?)
                }
                32 => {
                    let samples: Result<Vec<f32>, _> = reader
                        .samples::<i32>()
                        .map(|s| s.map(|v| v as f32 / 2147483648.0))
                        .collect();
                    Ok(samples.context("Failed to read i32 samples")?)
                }
                bits => Err(anyhow::anyhow!("Unsupported WAV bit depth: {}", bits)),
            }
        }
    }
}

/// Downmix interleaved multi-channel audio to mono by averaging channels.
fn downmix_to_mono(samples: &[f32], channels: u16) -> Vec<f32> {
    let ch = channels as usize;
    samples
        .chunks_exact(ch)
        .map(|frame| frame.iter().sum::<f32>() / ch as f32)
        .collect()
}

/// Resample mono audio via linear interpolation.
///
/// For merging two audio streams that may differ in sample rate (e.g. 44100 vs 48000),
/// linear interpolation provides adequate quality without external dependencies.
fn resample_mono(samples: &[f32], from_rate: u32, to_rate: u32) -> Result<Vec<f32>> {
    if from_rate == to_rate {
        return Ok(samples.to_vec());
    }

    let ratio = from_rate as f64 / to_rate as f64;
    let out_len = ((samples.len() as f64) / ratio).ceil() as usize;
    let mut output = Vec::with_capacity(out_len);

    for i in 0..out_len {
        let src_pos = i as f64 * ratio;
        let idx = src_pos as usize;
        let frac = (src_pos - idx as f64) as f32;

        let sample = if idx + 1 < samples.len() {
            samples[idx] * (1.0 - frac) + samples[idx + 1] * frac
        } else if idx < samples.len() {
            samples[idx]
        } else {
            0.0
        };
        output.push(sample);
    }

    Ok(output)
}
