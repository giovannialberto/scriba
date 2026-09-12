//! Owner voice enrollment: turn a clip of the owner speaking into stored
//! voice samples that diarization can recognize.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

use super::diarization::{self, VoiceSample};
use super::transcription::{find_ffmpeg, get_audio_channels};
use crate::database::Database;

/// Convert any audio file to a 16 kHz mono WAV next to it (temporary).
/// Two-track recordings contribute only their mic side.
pub fn to_mono_16k(input: &Path) -> Result<PathBuf> {
    let out = input
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join("_tmp_voice_16k.wav");
    let ffmpeg = find_ffmpeg()?;
    let two_track = get_audio_channels(input).map(|c| c >= 2).unwrap_or(false);
    let mut cmd = std::process::Command::new(&ffmpeg);
    cmd.args(["-y", "-loglevel", "error", "-i"]).arg(input);
    if two_track {
        cmd.args(["-af", "pan=mono|c0=c0"]);
    }
    cmd.args(["-ar", "16000", "-ac", "1", "-f", "wav"])
        .arg(&out);
    let status = cmd.status().context("Failed to run ffmpeg")?;
    if !status.success() {
        anyhow::bail!("ffmpeg could not convert {}", input.display());
    }
    Ok(out)
}

/// Embed a clip of the owner speaking into voice samples.
pub async fn owner_samples_from_file(input: &Path) -> Result<Vec<VoiceSample>> {
    let models = diarization::ensure_diarization_models(true).await?;
    let wav = to_mono_16k(input)?;
    let result = diarization::embed_enrollment_clip(&wav, &models);
    let _ = std::fs::remove_file(&wav);
    let samples = result?;
    if samples.is_empty() {
        anyhow::bail!("No speech found in {}", input.display());
    }
    Ok(samples)
}

/// Store owner voice samples from `input`. Returns how many were stored.
pub async fn enroll_owner_from_file(input: &Path, db: &mut Database) -> Result<usize> {
    let samples = owner_samples_from_file(input).await?;
    store_owner_samples(db, &samples, "enrollment")?;
    Ok(samples.len())
}

/// Persist owner samples and keep the table bounded.
pub fn store_owner_samples(db: &mut Database, samples: &[VoiceSample], source: &str) -> Result<()> {
    for s in samples {
        db.add_speaker_sample("owner", true, &s.embedding, s.duration_secs, source, diarization::EMBEDDING_MODEL_ID, None)?;
    }
    db.prune_speaker_samples("owner", diarization::OWNER_SAMPLES_KEPT, diarization::EMBEDDING_MODEL_ID)?;
    Ok(())
}
