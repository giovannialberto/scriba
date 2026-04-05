//! Speaker diarization for Scriba.
//!
//! Uses sherpa-onnx's offline speaker diarization to identify distinct speakers
//! in audio, then merges speaker labels with timestamped transcript segments.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use sherpa_onnx::Wave;
use std::path::{Path, PathBuf};

use crate::utils::BASE_PATH;

// Re-export model paths for external async download
pub use self::models::{ensure_diarization_models, DiarizationModelPaths};

/// A whisper segment with timestamp and text.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimedSegment {
    pub start: f64, // seconds
    pub end: f64,   // seconds
    pub text: String,
}

/// A speaker turn from diarization.
#[derive(Debug, Clone)]
pub struct SpeakerTurn {
    pub start: f64,
    pub end: f64,
    pub speaker: usize, // generic speaker index
}

/// A transcript segment attributed to a speaker.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiarizedSegment {
    pub start: f64,
    pub end: f64,
    pub speaker: String, // "Speaker 0" or resolved name like "Giovanni"
    pub text: String,
}

/// Complete diarization result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiarizedTranscript {
    pub segments: Vec<DiarizedSegment>,
    pub speakers: Vec<String>, // ordered list of speaker labels
}

/// Model download and management (async-safe).
mod models {
    use super::*;
    use futures_util::StreamExt;
    use std::io::Write as _;
    use std::process::Command;

    /// Paths to the diarization models needed by sherpa-onnx.
    #[derive(Debug, Clone)]
    pub struct DiarizationModelPaths {
        pub segmentation: PathBuf,
        pub embedding: PathBuf,
    }

    /// Get the path to diarization models directory.
    fn models_dir() -> PathBuf {
        BASE_PATH.join("models").join("sherpa").join("diarization")
    }

    /// Ensure diarization models are downloaded (async-safe).
    ///
    /// Must be called from an async context before `diarize_audio()`.
    pub async fn ensure_diarization_models() -> Result<DiarizationModelPaths> {
        let dir = models_dir();
        std::fs::create_dir_all(&dir).ok();

        // Segmentation model (pyannote segmentation 3.0 for sherpa-onnx)
        let seg_path = dir.join("sherpa-onnx-pyannote-segmentation-3.0.onnx");
        if !seg_path.exists() {
            let url = "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3.0.tar.bz2";
            let archive = dir.join("sherpa-onnx-pyannote-segmentation-3.0.tar.bz2");
            download_model_async(url, &archive).await
                .with_context(|| format!("Failed to download segmentation model from {}", url))?;
            extract_archive(&archive, &dir)?;
            let _ = std::fs::remove_file(&archive);
        }

        // Embedding model (3D-Speaker for speaker identification)
        let emb_path = dir.join("3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx");
        if !emb_path.exists() {
            let url = "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx";
            download_model_async(url, &emb_path).await
                .with_context(|| format!("Failed to download embedding model from {}", url))?;
        }

        Ok(DiarizationModelPaths {
            segmentation: seg_path,
            embedding: emb_path,
        })
    }

    fn extract_archive(archive: &Path, dest_dir: &Path) -> Result<()> {
        let output = Command::new("tar")
            .args(["xjf", archive.to_string_lossy().as_ref()])
            .current_dir(dest_dir)
            .output()
            .context("Failed to extract model archive (tar not found?)")?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(anyhow::anyhow!("Failed to extract model archive: {}", stderr));
        }
        Ok(())
    }

    /// Download a file using async reqwest (safe inside tokio runtime).
    async fn download_model_async(url: &str, dest: &Path) -> Result<()> {
        let client = reqwest::Client::new();
        let resp = client
            .get(url)
            .send()
            .await
            .with_context(|| format!("Failed to fetch {}", url))?
            .error_for_status()
            .with_context(|| format!("HTTP error downloading {}", url))?;

        let total = resp.content_length();
        let mut stream = resp.bytes_stream();
        let mut file = std::fs::File::create(dest)
            .with_context(|| format!("Failed to create {}", dest.display()))?;
        let mut downloaded: u64 = 0;

        while let Some(chunk) = stream.next().await {
            let chunk = chunk.context("Error reading download stream")?;
            std::io::Write::write_all(&mut file, &chunk)
                .context("Failed to write model data")?;
            downloaded += chunk.len() as u64;
            if let Some(total) = total {
                if downloaded % (5 * 1024 * 1024) < chunk.len() as u64 {
                    let pct = (downloaded as f64 / total as f64) * 100.0;
                    eprint!("\rDownloading diarization model... {:.1}%", pct);
                    let _ = std::io::stderr().flush();
                }
            }
        }
        if total.is_some() {
            eprintln!();
        }

        Ok(())
    }
}

/// Run speaker diarization on a 16kHz mono WAV file using sherpa-onnx.
///
/// Model paths must be provided (use `ensure_diarization_models()` from async
/// code to download them before calling this sync function).
///
/// Returns a list of speaker turns with timestamps and speaker indices.
pub fn diarize_audio(
    wav_path: &Path,
    max_speakers: usize,
    model_paths: &DiarizationModelPaths,
) -> Result<Vec<SpeakerTurn>> {
    let wave = Wave::read(wav_path.to_string_lossy().as_ref())
        .ok_or_else(|| anyhow::anyhow!("Failed to read WAV file for diarization: {}", wav_path.display()))?;

    let mut config = sherpa_onnx::OfflineSpeakerDiarizationConfig::default();
    config.segmentation.pyannote.model =
        Some(model_paths.segmentation.to_string_lossy().into_owned());
    config.embedding.model =
        Some(model_paths.embedding.to_string_lossy().into_owned());
    if max_speakers > 0 {
        config.clustering.num_clusters = max_speakers as i32;
    }
    config.segmentation.num_threads = 2;
    config.embedding.num_threads = 2;

    let diarizer = sherpa_onnx::OfflineSpeakerDiarization::create(&config)
        .ok_or_else(|| anyhow::anyhow!("Failed to create sherpa-onnx speaker diarizer. Check model files."))?;

    let result = diarizer.process(wave.samples())
        .ok_or_else(|| anyhow::anyhow!("Speaker diarization failed"))?;

    let segments = result.sort_by_start_time();
    let mut turns = Vec::new();
    for segment in &segments {
        turns.push(SpeakerTurn {
            start: segment.start as f64,
            end: segment.end as f64,
            speaker: segment.speaker as usize,
        });
    }

    Ok(turns)
}

/// Merge Whisper timed segments with speaker turns by temporal overlap.
///
/// For each whisper segment, finds the speaker turn with the maximum
/// temporal overlap and assigns that speaker.
pub fn merge_segments(
    whisper_segments: &[TimedSegment],
    speaker_turns: &[SpeakerTurn],
) -> Vec<DiarizedSegment> {
    whisper_segments
        .iter()
        .map(|ws| {
            let speaker_label = find_best_speaker(ws.start, ws.end, speaker_turns);
            DiarizedSegment {
                start: ws.start,
                end: ws.end,
                speaker: speaker_label,
                text: ws.text.clone(),
            }
        })
        .collect()
}

/// Find the speaker with maximum temporal overlap for a given time range.
fn find_best_speaker(start: f64, end: f64, turns: &[SpeakerTurn]) -> String {
    let mut best_speaker: Option<usize> = None;
    let mut best_overlap: f64 = 0.0;

    for turn in turns {
        let overlap_start = start.max(turn.start);
        let overlap_end = end.min(turn.end);
        let overlap = (overlap_end - overlap_start).max(0.0);

        if overlap > best_overlap {
            best_overlap = overlap;
            best_speaker = Some(turn.speaker);
        }
    }

    match best_speaker {
        Some(idx) => format!("Speaker {}", idx),
        None => "Unknown".to_string(),
    }
}

/// Build a DiarizedTranscript from merged segments.
pub fn build_diarized_transcript(segments: Vec<DiarizedSegment>) -> DiarizedTranscript {
    let mut speakers: Vec<String> = Vec::new();
    for seg in &segments {
        if !speakers.contains(&seg.speaker) {
            speakers.push(seg.speaker.clone());
        }
    }

    DiarizedTranscript { segments, speakers }
}

/// Format a diarized transcript as plain text with speaker labels.
///
/// Output format: `[Speaker Name] segment text`
/// Adjacent segments from the same speaker are merged.
pub fn format_diarized_text(transcript: &DiarizedTranscript) -> String {
    if transcript.segments.is_empty() {
        return String::new();
    }

    let mut lines: Vec<String> = Vec::new();
    let mut current_speaker = String::new();
    let mut current_text = String::new();

    for seg in &transcript.segments {
        if seg.speaker == current_speaker {
            // Same speaker, append text
            if !current_text.is_empty() {
                current_text.push(' ');
            }
            current_text.push_str(seg.text.trim());
        } else {
            // New speaker, flush previous
            if !current_text.is_empty() {
                lines.push(format!("[{}] {}", current_speaker, current_text));
            }
            current_speaker = seg.speaker.clone();
            current_text = seg.text.trim().to_string();
        }
    }

    // Flush last segment
    if !current_text.is_empty() {
        lines.push(format!("[{}] {}", current_speaker, current_text));
    }

    lines.join("\n")
}

/// Apply speaker name resolution to a diarized transcript.
///
/// Takes a mapping from generic labels ("Speaker 0") to resolved names ("Giovanni").
/// Entries with None values are kept as-is.
pub fn apply_speaker_names(
    transcript: &mut DiarizedTranscript,
    name_map: &std::collections::HashMap<String, Option<String>>,
) {
    for seg in &mut transcript.segments {
        if let Some(Some(resolved)) = name_map.get(&seg.speaker) {
            seg.speaker = resolved.clone();
        }
    }

    // Rebuild speakers list
    let mut speakers: Vec<String> = Vec::new();
    for seg in &transcript.segments {
        if !speakers.contains(&seg.speaker) {
            speakers.push(seg.speaker.clone());
        }
    }
    transcript.speakers = speakers;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_merge_segments_basic() {
        let whisper = vec![
            TimedSegment { start: 0.0, end: 3.0, text: "Hello everyone".to_string() },
            TimedSegment { start: 3.0, end: 6.0, text: "Hi there".to_string() },
            TimedSegment { start: 6.0, end: 10.0, text: "Let's begin".to_string() },
        ];

        let turns = vec![
            SpeakerTurn { start: 0.0, end: 4.0, speaker: 0 },
            SpeakerTurn { start: 4.0, end: 10.0, speaker: 1 },
        ];

        let result = merge_segments(&whisper, &turns);

        assert_eq!(result.len(), 3);
        assert_eq!(result[0].speaker, "Speaker 0");
        assert_eq!(result[0].text, "Hello everyone");
        // "Hi there" (3.0-6.0) overlaps with speaker 0 (3.0-4.0) = 1s and speaker 1 (4.0-6.0) = 2s
        assert_eq!(result[1].speaker, "Speaker 1");
        assert_eq!(result[2].speaker, "Speaker 1");
    }

    #[test]
    fn test_merge_segments_no_turns() {
        let whisper = vec![
            TimedSegment { start: 0.0, end: 3.0, text: "Hello".to_string() },
        ];

        let result = merge_segments(&whisper, &[]);

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].speaker, "Unknown");
    }

    #[test]
    fn test_merge_segments_no_overlap() {
        let whisper = vec![
            TimedSegment { start: 100.0, end: 110.0, text: "Late segment".to_string() },
        ];

        let turns = vec![
            SpeakerTurn { start: 0.0, end: 5.0, speaker: 0 },
        ];

        let result = merge_segments(&whisper, &turns);

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].speaker, "Unknown");
    }

    #[test]
    fn test_format_diarized_text() {
        let transcript = DiarizedTranscript {
            segments: vec![
                DiarizedSegment { start: 0.0, end: 3.0, speaker: "Giovanni".to_string(), text: "Hello everyone".to_string() },
                DiarizedSegment { start: 3.0, end: 5.0, speaker: "Giovanni".to_string(), text: "welcome to the meeting".to_string() },
                DiarizedSegment { start: 5.0, end: 8.0, speaker: "Sara".to_string(), text: "Thanks for having me".to_string() },
            ],
            speakers: vec!["Giovanni".to_string(), "Sara".to_string()],
        };

        let text = format_diarized_text(&transcript);
        assert_eq!(text, "[Giovanni] Hello everyone welcome to the meeting\n[Sara] Thanks for having me");
    }

    #[test]
    fn test_format_diarized_text_empty() {
        let transcript = DiarizedTranscript {
            segments: vec![],
            speakers: vec![],
        };

        assert_eq!(format_diarized_text(&transcript), "");
    }

    #[test]
    fn test_apply_speaker_names() {
        let mut transcript = DiarizedTranscript {
            segments: vec![
                DiarizedSegment { start: 0.0, end: 3.0, speaker: "Speaker 0".to_string(), text: "Hello".to_string() },
                DiarizedSegment { start: 3.0, end: 6.0, speaker: "Speaker 1".to_string(), text: "Hi".to_string() },
                DiarizedSegment { start: 6.0, end: 9.0, speaker: "Speaker 0".to_string(), text: "Let's go".to_string() },
            ],
            speakers: vec!["Speaker 0".to_string(), "Speaker 1".to_string()],
        };

        let mut name_map = std::collections::HashMap::new();
        name_map.insert("Speaker 0".to_string(), Some("Giovanni".to_string()));
        name_map.insert("Speaker 1".to_string(), None); // unresolved

        apply_speaker_names(&mut transcript, &name_map);

        assert_eq!(transcript.segments[0].speaker, "Giovanni");
        assert_eq!(transcript.segments[1].speaker, "Speaker 1"); // unchanged
        assert_eq!(transcript.segments[2].speaker, "Giovanni");
        assert_eq!(transcript.speakers, vec!["Giovanni", "Speaker 1"]);
    }

    #[test]
    fn test_build_diarized_transcript() {
        let segments = vec![
            DiarizedSegment { start: 0.0, end: 3.0, speaker: "Speaker 0".to_string(), text: "A".to_string() },
            DiarizedSegment { start: 3.0, end: 6.0, speaker: "Speaker 1".to_string(), text: "B".to_string() },
            DiarizedSegment { start: 6.0, end: 9.0, speaker: "Speaker 0".to_string(), text: "C".to_string() },
        ];

        let transcript = build_diarized_transcript(segments);

        assert_eq!(transcript.speakers, vec!["Speaker 0", "Speaker 1"]);
        assert_eq!(transcript.segments.len(), 3);
    }
}
