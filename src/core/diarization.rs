//! Speaker diarization on top of the local transcriber: who spoke when.
//!
//! The transcriber already cuts the recording into speech segments (Silero
//! VAD), and a speaker rarely changes inside one of them. So instead of a
//! sliding-window segmentation model, each transcript segment gets one speaker
//! embedding (sherpa-onnx, 3D-Speaker ERes2Net) and the embeddings are
//! clustered by cosine similarity. That runs in seconds for an hour of audio.
//!
//! Two-track meeting recordings (mic left, system audio right) get the owner
//! for free: a segment where the mic carries clearly more energy than the
//! system track is the owner speaking, and only the other segments are
//! clustered, on the system track. Single-track recordings are clustered whole
//! and labelled "Speaker N".

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use sherpa_onnx::{SpeakerEmbeddingExtractor, SpeakerEmbeddingExtractorConfig};

use super::transcription::download_file_streaming;
use crate::utils::BASE_PATH;

const EMBEDDING_FILE: &str = "3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx";
const EMBEDDING_URL: &str = "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx";

/// Cosine similarity above which two speaker clusters are merged.
pub const DEFAULT_SIMILARITY_THRESHOLD: f32 = 0.7;
/// Segments shorter than this have unreliable embeddings; they inherit the
/// speaker of the closest neighbouring segment instead.
pub const MIN_EMBED_SECS: f32 = 0.8;
/// The mic must carry this much more RMS energy than the system track for a
/// segment to count as the owner speaking (covers speaker bleed into the mic).
pub const OWNER_ENERGY_RATIO: f32 = 2.0;
/// Below this RMS a track is considered silent for the segment.
const SILENCE_RMS: f32 = 1e-4;
/// Left/right correlation above which a "stereo" file is really a mono mix.
const IDENTICAL_TRACKS_CORRELATION: f32 = 0.98;
/// Cosine similarity above which a cluster is recognized as a known voice.
pub const PROFILE_MATCH_THRESHOLD: f32 = 0.62;
/// Owner voice samples harvested per meeting from the mic track.
pub const OWNER_SAMPLES_PER_RECORDING: usize = 8;
/// Newest owner samples kept in the database.
pub const OWNER_SAMPLES_KEPT: usize = 80;
/// Minimum length of a segment used as a voice sample.
pub const MIN_SAMPLE_SECS: f32 = 2.0;
/// Enrollment clips are cut into windows of this length before embedding.
pub const ENROLLMENT_WINDOW_SECS: f32 = 3.0;
/// Enrollment windows quieter than this RMS are skipped (silence, breaths).
const ENROLLMENT_MIN_RMS: f32 = 0.01;

/// A cluster with less speech than this (or than `MIN_CLUSTER_SHARE` of all
/// speech, whichever is smaller) is absorbed into its most similar neighbour:
/// it is far more likely a bad embedding than a real participant.
const MIN_CLUSTER_SECS: f32 = 6.0;
const MIN_CLUSTER_SHARE: f32 = 0.05;

/// Where the speaker embedding model lives once downloaded.
#[derive(Debug, Clone)]
pub struct DiarizationModels {
    pub embedding: PathBuf,
}

fn models_dir() -> PathBuf {
    BASE_PATH.join("models").join("sherpa")
}

/// Whether the model is present (no download needed).
pub fn models_downloaded() -> bool {
    models_dir().join(EMBEDDING_FILE).exists()
}

/// Download the speaker embedding model on first use (about 40 MB).
pub async fn ensure_diarization_models(quiet: bool) -> Result<DiarizationModels> {
    let dir = models_dir();
    std::fs::create_dir_all(&dir).ok();
    let embedding = dir.join(EMBEDDING_FILE);
    if !embedding.exists() {
        download_file_streaming(EMBEDDING_URL, &embedding, quiet)
            .await
            .context("Failed to download the speaker embedding model")?;
    }
    Ok(DiarizationModels { embedding })
}

/// A transcribed stretch of speech with its position in the recording.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TranscriptSegment {
    pub start: f32,
    pub end: f32,
    pub text: String,
}

/// A transcribed segment with a speaker label, as stored in the database.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LabelledSegment {
    pub start: f32,
    pub end: f32,
    pub speaker: String,
    pub text: String,
}

/// A voice Scriba already knows, ready for matching.
#[derive(Debug, Clone)]
pub struct KnownSpeaker {
    /// Label to use in transcripts (owner's name or entity name).
    pub name: String,
    pub is_owner: bool,
    /// L2-normalized centroid embedding.
    pub centroid: Vec<f32>,
}

impl KnownSpeaker {
    pub fn similarity(&self, embedding: &[f32]) -> f32 {
        cosine(&self.centroid, embedding)
    }
}

/// A voice sample worth remembering: embedding plus how much speech backed it.
#[derive(Debug, Clone)]
pub struct VoiceSample {
    pub embedding: Vec<f32>,
    pub duration_secs: f32,
}

/// Result of diarizing a transcript.
#[derive(Debug, Clone)]
pub struct Diarized {
    /// Speaker-labelled transcript text ("Name: what they said" lines).
    pub text: String,
    pub segments: Vec<LabelledSegment>,
    /// Distinct speaker labels in order of first appearance.
    pub speakers: Vec<String>,
    /// Owner voice samples taken from the mic track (two-track recordings only).
    pub owner_samples: Vec<VoiceSample>,
}

/// Tunables, taken from `DiarizationConfig`.
#[derive(Debug, Clone, Copy)]
pub struct DiarizationOptions {
    pub similarity_threshold: f32,
    pub max_speakers: usize,
}

impl Default for DiarizationOptions {
    fn default() -> Self {
        Self {
            similarity_threshold: DEFAULT_SIMILARITY_THRESHOLD,
            max_speakers: 6,
        }
    }
}

// ─── Embeddings ──────────────────────────────────────────────────────────────

/// Speaker embedding model, loaded once per recording.
pub struct SpeakerEmbedder {
    extractor: SpeakerEmbeddingExtractor,
}

impl SpeakerEmbedder {
    pub fn load(models: &DiarizationModels) -> Result<Self> {
        let config = SpeakerEmbeddingExtractorConfig {
            model: Some(models.embedding.to_string_lossy().into_owned()),
            num_threads: std::thread::available_parallelism()
                .map(|n| (n.get() / 2).clamp(1, 4) as i32)
                .unwrap_or(2),
            ..Default::default()
        };
        let extractor = SpeakerEmbeddingExtractor::create(&config)
            .ok_or_else(|| anyhow::anyhow!("Failed to load the speaker embedding model"))?;
        Ok(Self { extractor })
    }

    /// L2-normalized embedding of one stretch of speech, or `None` when too short.
    pub fn embed(&self, samples: &[f32], rate: u32) -> Option<Vec<f32>> {
        if (samples.len() as f32) < MIN_EMBED_SECS * rate as f32 {
            return None;
        }
        let stream = self.extractor.create_stream()?;
        stream.accept_waveform(rate as i32, samples);
        stream.input_finished();
        if !self.extractor.is_ready(&stream) {
            return None;
        }
        let mut emb = self.extractor.compute(&stream)?;
        let norm = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm <= 0.0 {
            return None;
        }
        emb.iter_mut().for_each(|x| *x /= norm);
        Some(emb)
    }

    /// One embedding per transcript segment, sliced from `samples`.
    pub fn embed_segments(
        &self,
        samples: &[f32],
        rate: u32,
        transcript: &[TranscriptSegment],
    ) -> Vec<Option<Vec<f32>>> {
        transcript
            .iter()
            .map(|t| self.embed(window(samples, t.start, t.end, rate), rate))
            .collect()
    }
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Agglomerative clustering (average linkage on cosine similarity) of the
/// embedded segments. Returns one cluster id per segment; segments without an
/// embedding get `None`. Clusters merge while their similarity is at least
/// `threshold`, then further until at most `max_speakers` remain, and finally
/// clusters with too little speech are absorbed.
///
/// Centroid sums and the pairwise similarity table are updated incrementally,
/// so an hour of audio (a few hundred segments) clusters in well under a second.
pub fn cluster_speakers(
    embeddings: &[Option<Vec<f32>>],
    durations: &[f32],
    options: DiarizationOptions,
) -> Vec<Option<usize>> {
    let members: Vec<usize> = (0..embeddings.len())
        .filter(|&i| embeddings[i].is_some())
        .collect();
    if members.is_empty() {
        return vec![None; embeddings.len()];
    }
    let max_speakers = options.max_speakers.max(1);

    // Cluster state: member indices, unnormalized centroid sum, total speech.
    let mut clusters: Vec<Vec<usize>> = members.iter().map(|&i| vec![i]).collect();
    let mut sums: Vec<Vec<f32>> = members
        .iter()
        .map(|&i| embeddings[i].clone().unwrap())
        .collect();
    let mut secs: Vec<f32> = members
        .iter()
        .map(|&i| durations.get(i).copied().unwrap_or(0.0))
        .collect();
    let normalized = |v: &[f32]| -> Vec<f32> {
        let n = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
        v.iter().map(|x| x / n).collect()
    };
    let mut centroids: Vec<Vec<f32>> = sums.iter().map(|v| normalized(v)).collect();
    let k = clusters.len();
    let mut sim: Vec<Vec<f32>> = vec![vec![0.0; k]; k];
    for i in 0..k {
        for j in (i + 1)..k {
            let v = cosine(&centroids[i], &centroids[j]);
            sim[i][j] = v;
            sim[j][i] = v;
        }
    }
    let mut alive: Vec<bool> = vec![true; k];
    let mut alive_count = k;

    let merge = |i: usize,
                 j: usize,
                 clusters: &mut Vec<Vec<usize>>,
                 sums: &mut Vec<Vec<f32>>,
                 secs: &mut Vec<f32>,
                 centroids: &mut Vec<Vec<f32>>,
                 sim: &mut Vec<Vec<f32>>,
                 alive: &mut Vec<bool>| {
        let moved = std::mem::take(&mut clusters[j]);
        clusters[i].extend(moved);
        let added = std::mem::take(&mut sums[j]);
        for (a, b) in sums[i].iter_mut().zip(added) {
            *a += b;
        }
        secs[i] += secs[j];
        secs[j] = 0.0;
        alive[j] = false;
        centroids[i] = normalized(&sums[i]);
        for x in 0..alive.len() {
            if alive[x] && x != i {
                let v = cosine(&centroids[i], &centroids[x]);
                sim[i][x] = v;
                sim[x][i] = v;
            }
        }
    };

    // Phase 1: merge the most similar pair while above threshold or over the cap.
    while alive_count > 1 {
        let mut best: Option<(f32, usize, usize)> = None;
        for i in 0..k {
            if !alive[i] {
                continue;
            }
            for j in (i + 1)..k {
                if alive[j] && best.is_none_or(|(b, _, _)| sim[i][j] > b) {
                    best = Some((sim[i][j], i, j));
                }
            }
        }
        let Some((best_sim, i, j)) = best else { break };
        if best_sim < options.similarity_threshold && alive_count <= max_speakers {
            break;
        }
        merge(
            i,
            j,
            &mut clusters,
            &mut sums,
            &mut secs,
            &mut centroids,
            &mut sim,
            &mut alive,
        );
        alive_count -= 1;
    }

    // Phase 2: absorb clusters too small to be a real participant.
    let total_secs: f32 = secs.iter().sum();
    let min_secs = MIN_CLUSTER_SECS.min(total_secs * MIN_CLUSTER_SHARE);
    while alive_count > 1 {
        let small = (0..k)
            .filter(|&i| alive[i] && secs[i] < min_secs)
            .min_by(|&a, &b| secs[a].total_cmp(&secs[b]));
        let Some(i) = small else { break };
        let target = (0..k)
            .filter(|&x| alive[x] && x != i)
            .max_by(|&a, &b| sim[i][a].total_cmp(&sim[i][b]));
        let Some(target) = target else { break };
        merge(
            target,
            i,
            &mut clusters,
            &mut sums,
            &mut secs,
            &mut centroids,
            &mut sim,
            &mut alive,
        );
        alive_count -= 1;
    }

    let mut assignment = vec![None; embeddings.len()];
    let mut next_id = 0;
    for (c, members) in clusters.iter().enumerate() {
        if !alive[c] {
            continue;
        }
        for &m in members {
            assignment[m] = Some(next_id);
        }
        next_id += 1;
    }
    assignment
}

/// Fill unassigned (too short) segments with the speaker of the nearest
/// assigned neighbour in time, preferring the previous one.
pub fn fill_gaps_by_neighbour(assignment: &mut [Option<usize>]) {
    let n = assignment.len();
    for i in 0..n {
        if assignment[i].is_some() {
            continue;
        }
        let prev = (0..i).rev().find_map(|j| assignment[j].map(|a| (i - j, a)));
        let next = ((i + 1)..n).find_map(|j| assignment[j].map(|a| (j - i, a)));
        assignment[i] = match (prev, next) {
            (Some((dp, a)), Some((dn, b))) => Some(if dp <= dn { a } else { b }),
            (Some((_, a)), None) | (None, Some((_, a))) => Some(a),
            (None, None) => None,
        };
    }
}

// ─── Two-track helpers ───────────────────────────────────────────────────────

/// Read a WAV as f32 samples per channel.
pub fn read_wav_channels(wav: &Path) -> Result<(Vec<Vec<f32>>, u32)> {
    let mut reader = hound::WavReader::open(wav).context("Failed to open WAV")?;
    let spec = reader.spec();
    let interleaved: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().collect::<Result<_, _>>()?,
        hound::SampleFormat::Int => match spec.bits_per_sample {
            16 => reader
                .samples::<i16>()
                .map(|s| s.map(|v| v as f32 / 32768.0))
                .collect::<Result<_, _>>()?,
            32 => reader
                .samples::<i32>()
                .map(|s| s.map(|v| v as f32 / 2147483648.0))
                .collect::<Result<_, _>>()?,
            bits => anyhow::bail!("Unsupported WAV bit depth: {bits}"),
        },
    };
    let ch = spec.channels as usize;
    let channels = (0..ch)
        .map(|c| interleaved.iter().skip(c).step_by(ch).copied().collect())
        .collect();
    Ok((channels, spec.sample_rate))
}

fn rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    (samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32).sqrt()
}

fn window(samples: &[f32], start: f32, end: f32, rate: u32) -> &[f32] {
    let a = ((start.max(0.0) * rate as f32) as usize).min(samples.len());
    let b = ((end.max(0.0) * rate as f32) as usize).clamp(a, samples.len());
    &samples[a..b]
}

/// Pearson correlation of two tracks on a sparse sample; near 1.0 means the
/// "stereo" file is really the same mix on both sides.
pub fn tracks_correlation(left: &[f32], right: &[f32]) -> f32 {
    let n = left.len().min(right.len());
    if n < 2 {
        return 1.0;
    }
    let step = (n / 200_000).max(1);
    let (mut sl, mut sr, mut sll, mut srr, mut slr, mut count) =
        (0f64, 0f64, 0f64, 0f64, 0f64, 0f64);
    for i in (0..n).step_by(step) {
        let (l, r) = (left[i] as f64, right[i] as f64);
        sl += l;
        sr += r;
        sll += l * l;
        srr += r * r;
        slr += l * r;
        count += 1.0;
    }
    let cov = slr / count - (sl / count) * (sr / count);
    let vl = sll / count - (sl / count).powi(2);
    let vr = srr / count - (sr / count).powi(2);
    if vl <= 0.0 || vr <= 0.0 {
        return 1.0;
    }
    (cov / (vl * vr).sqrt()) as f32
}

/// For each transcript segment, whether the mic (left) dominates the system
/// track (right) by `ratio`, i.e. the owner is speaking.
pub fn owner_mask(
    left: &[f32],
    right: &[f32],
    rate: u32,
    transcript: &[TranscriptSegment],
    ratio: f32,
) -> Vec<bool> {
    transcript
        .iter()
        .map(|t| {
            let l = rms(window(left, t.start, t.end, rate));
            let r = rms(window(right, t.start, t.end, rate));
            l > SILENCE_RMS && l > r * ratio
        })
        .collect()
}

// ─── Labelling ───────────────────────────────────────────────────────────────

/// Render labelled segments as "Speaker: text" lines, merging consecutive
/// segments from the same speaker.
pub fn render_transcript(segments: &[LabelledSegment]) -> String {
    let mut lines: Vec<(String, String)> = Vec::new();
    for seg in segments {
        let text = seg.text.trim();
        if text.is_empty() {
            continue;
        }
        match lines.last_mut() {
            Some((speaker, body)) if *speaker == seg.speaker => {
                body.push(' ');
                body.push_str(text);
            }
            _ => lines.push((seg.speaker.clone(), text.to_string())),
        }
    }
    lines
        .into_iter()
        .map(|(speaker, body)| format!("{speaker}: {body}"))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Distinct speaker labels in order of first appearance.
pub fn speaker_names(segments: &[LabelledSegment]) -> Vec<String> {
    let mut names: Vec<String> = Vec::new();
    for seg in segments {
        if !names.contains(&seg.speaker) {
            names.push(seg.speaker.clone());
        }
    }
    names
}

/// Recognize clusters: for each cluster id, the best-matching known speaker
/// above `PROFILE_MATCH_THRESHOLD`, by centroid similarity.
pub fn match_clusters(
    assignment: &[Option<usize>],
    embeddings: &[Option<Vec<f32>>],
    known: &[KnownSpeaker],
) -> std::collections::HashMap<usize, String> {
    let mut matched = std::collections::HashMap::new();
    if known.is_empty() {
        return matched;
    }
    let mut ids: Vec<usize> = assignment.iter().flatten().copied().collect();
    ids.sort_unstable();
    ids.dedup();
    let dim = embeddings.iter().flatten().next().map(|e| e.len()).unwrap_or(0);
    for id in ids {
        let mut sum = vec![0f32; dim];
        let mut n = 0;
        for (i, a) in assignment.iter().enumerate() {
            if *a == Some(id)
                && let Some(e) = &embeddings[i]
            {
                for (acc, v) in sum.iter_mut().zip(e) {
                    *acc += v;
                }
                n += 1;
            }
        }
        if n == 0 {
            continue;
        }
        let norm = sum.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
        sum.iter_mut().for_each(|x| *x /= norm);
        if let Some((best, sim)) = known
            .iter()
            .map(|k| (k, k.similarity(&sum)))
            .max_by(|a, b| a.1.total_cmp(&b.1))
            && sim >= PROFILE_MATCH_THRESHOLD
        {
            matched.insert(id, best.name.clone());
        }
    }
    matched
}

/// Owner voice samples from a two-track recording: the longest owner segments
/// on the mic track, embedded separately.
pub fn owner_samples(
    embedder: &SpeakerEmbedder,
    left: &[f32],
    rate: u32,
    transcript: &[TranscriptSegment],
    owner: &[bool],
) -> Vec<VoiceSample> {
    let mut candidates: Vec<&TranscriptSegment> = transcript
        .iter()
        .zip(owner)
        .filter(|(t, is_owner)| **is_owner && t.end - t.start >= MIN_SAMPLE_SECS)
        .map(|(t, _)| t)
        .collect();
    candidates.sort_by(|a, b| (b.end - b.start).total_cmp(&(a.end - a.start)));
    candidates
        .into_iter()
        .take(OWNER_SAMPLES_PER_RECORDING)
        .filter_map(|t| {
            embedder
                .embed(window(left, t.start, t.end, rate), rate)
                .map(|embedding| VoiceSample { embedding, duration_secs: t.end - t.start })
        })
        .collect()
}

/// Cut a clip into fixed windows and keep the ones with speech in them.
pub fn enrollment_windows(samples: &[f32], rate: u32) -> Vec<(usize, usize)> {
    let win = (ENROLLMENT_WINDOW_SECS * rate as f32) as usize;
    if win == 0 {
        return Vec::new();
    }
    let mut out = Vec::new();
    let mut start = 0;
    while start + win / 2 <= samples.len() {
        let end = (start + win).min(samples.len());
        if rms(&samples[start..end]) >= ENROLLMENT_MIN_RMS {
            out.push((start, end));
        }
        start += win;
    }
    out
}

/// Embed an enrollment clip (16 kHz mono WAV) into voice samples.
pub fn embed_enrollment_clip(wav_16k: &Path, models: &DiarizationModels) -> Result<Vec<VoiceSample>> {
    let (channels, rate) = read_wav_channels(wav_16k)?;
    let mono = channels.first().ok_or_else(|| anyhow::anyhow!("Empty enrollment clip"))?;
    let embedder = SpeakerEmbedder::load(models)?;
    Ok(enrollment_windows(mono, rate)
        .into_iter()
        .filter_map(|(a, b)| {
            embedder.embed(&mono[a..b], rate).map(|embedding| VoiceSample {
                embedding,
                duration_secs: (b - a) as f32 / rate as f32,
            })
        })
        .collect())
}

/// Map cluster ids to "Speaker N" labels numbered by first appearance,
/// starting at `first_number`.
fn cluster_labels(
    assignment: &[Option<usize>],
    first_number: usize,
    recognized: &std::collections::HashMap<usize, String>,
) -> Vec<String> {
    // Only unrecognized clusters consume "Speaker N" numbers.
    let mut order: Vec<usize> = Vec::new();
    for a in assignment.iter().flatten() {
        if !recognized.contains_key(a) && !order.contains(a) {
            order.push(*a);
        }
    }
    assignment
        .iter()
        .map(|a| match a {
            Some(c) => recognized.get(c).cloned().unwrap_or_else(|| {
                format!(
                    "Speaker {}",
                    first_number + order.iter().position(|o| o == c).unwrap_or(0)
                )
            }),
            None => "Speaker ?".to_string(),
        })
        .collect()
}

fn durations(transcript: &[TranscriptSegment]) -> Vec<f32> {
    transcript
        .iter()
        .map(|t| (t.end - t.start).max(0.0))
        .collect()
}

/// Label a single-track transcript from per-segment embeddings.
pub fn label_single_track(
    transcript: &[TranscriptSegment],
    embeddings: &[Option<Vec<f32>>],
    options: DiarizationOptions,
    known: &[KnownSpeaker],
) -> Vec<LabelledSegment> {
    let mut assignment = cluster_speakers(embeddings, &durations(transcript), options);
    let recognized = match_clusters(&assignment, embeddings, known);
    fill_gaps_by_neighbour(&mut assignment);
    let labels = cluster_labels(&assignment, 1, &recognized);
    transcript
        .iter()
        .zip(labels)
        .map(|(t, speaker)| LabelledSegment {
            start: t.start,
            end: t.end,
            speaker,
            text: t.text.clone(),
        })
        .collect()
}

/// Label a two-track transcript: owner segments by mic energy, the rest from
/// clustering the other participants' embeddings (system track).
pub fn label_two_track(
    transcript: &[TranscriptSegment],
    owner: &[bool],
    other_embeddings: &[Option<Vec<f32>>],
    options: DiarizationOptions,
    owner_name: &str,
    known: &[KnownSpeaker],
) -> Vec<LabelledSegment> {
    // Owner segments do not take part in clustering the others.
    let masked: Vec<Option<Vec<f32>>> = other_embeddings
        .iter()
        .zip(owner)
        .map(|(e, is_owner)| if *is_owner { None } else { e.clone() })
        .collect();
    let mut assignment = cluster_speakers(&masked, &durations(transcript), options);
    // Other participants may be voices we already know (never the owner here).
    let others_known: Vec<KnownSpeaker> = known.iter().filter(|k| !k.is_owner).cloned().collect();
    let recognized = match_clusters(&assignment, &masked, &others_known);
    let mut others_only: Vec<Option<usize>> = assignment
        .iter()
        .zip(owner)
        .map(|(a, is_owner)| if *is_owner { None } else { *a })
        .collect();
    fill_gaps_by_neighbour(&mut others_only);
    for (a, (filled, is_owner)) in assignment.iter_mut().zip(others_only.iter().zip(owner)) {
        if !*is_owner {
            *a = *filled;
        }
    }
    let labels = cluster_labels(
        &assignment
            .iter()
            .zip(owner)
            .map(|(a, is_owner)| if *is_owner { None } else { *a })
            .collect::<Vec<_>>(),
        2,
        &recognized,
    );
    transcript
        .iter()
        .enumerate()
        .map(|(i, t)| LabelledSegment {
            start: t.start,
            end: t.end,
            speaker: if owner[i] {
                owner_name.to_string()
            } else {
                labels[i].clone()
            },
            text: t.text.clone(),
        })
        .collect()
}

/// Attach speakers to a transcript.
///
/// `mono_16k` is the downmixed WAV the transcriber used; `stereo_16k` is the
/// same audio with both tracks kept, present only for two-track recordings.
/// `known` voices are recognized by centroid similarity.
pub fn diarize_transcript(
    transcript: &[TranscriptSegment],
    mono_16k: &Path,
    stereo_16k: Option<&Path>,
    models: &DiarizationModels,
    options: DiarizationOptions,
    owner_name: &str,
    known: &[KnownSpeaker],
) -> Result<Diarized> {
    let embedder = SpeakerEmbedder::load(models)?;
    let mut owner_voice: Vec<VoiceSample> = Vec::new();
    let segments = match stereo_16k {
        Some(stereo) => {
            let (channels, rate) = read_wav_channels(stereo)?;
            if channels.len() < 2 {
                anyhow::bail!("Expected a two-track WAV");
            }
            let (left, right) = (&channels[0], &channels[1]);
            if tracks_correlation(left, right) >= IDENTICAL_TRACKS_CORRELATION {
                // Plain stereo mix: nothing to learn from the channel split.
                let (mono, rate) = read_wav_channels(mono_16k)?;
                let embeddings = embedder.embed_segments(&mono[0], rate, transcript);
                label_single_track(transcript, &embeddings, options, known)
            } else {
                let owner = owner_mask(left, right, rate, transcript, OWNER_ENERGY_RATIO);
                let embeddings = embedder.embed_segments(right, rate, transcript);
                owner_voice = owner_samples(&embedder, left, rate, transcript, &owner);
                label_two_track(transcript, &owner, &embeddings, options, owner_name, known)
            }
        }
        None => {
            let (mono, rate) = read_wav_channels(mono_16k)?;
            let embeddings = embedder.embed_segments(&mono[0], rate, transcript);
            label_single_track(transcript, &embeddings, options, known)
        }
    };
    Ok(Diarized {
        text: render_transcript(&segments),
        speakers: speaker_names(&segments),
        segments,
        owner_samples: owner_voice,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(start: f32, end: f32, text: &str) -> TranscriptSegment {
        TranscriptSegment {
            start,
            end,
            text: text.into(),
        }
    }

    fn unit(v: &[f32]) -> Option<Vec<f32>> {
        let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        Some(v.iter().map(|x| x / n).collect())
    }

    #[test]
    fn clustering_groups_similar_embeddings_and_caps_speakers() {
        // Two clear voices along different axes, one noisy variant of each.
        let embeddings = vec![
            unit(&[1.0, 0.0, 0.0]),
            unit(&[0.95, 0.05, 0.0]),
            unit(&[0.0, 1.0, 0.0]),
            unit(&[0.05, 0.95, 0.0]),
            None,
        ];
        let durations = vec![50.0, 50.0, 50.0, 50.0, 0.4];
        let options = DiarizationOptions {
            similarity_threshold: 0.7,
            max_speakers: 6,
        };
        let mut assignment = cluster_speakers(&embeddings, &durations, options);
        assert_eq!(assignment[0], assignment[1]);
        assert_eq!(assignment[2], assignment[3]);
        assert_ne!(assignment[0], assignment[2]);
        assert_eq!(assignment[4], None);
        fill_gaps_by_neighbour(&mut assignment);
        assert_eq!(
            assignment[4], assignment[3],
            "short segment inherits its neighbour"
        );

        // A cap of one speaker forces everything together.
        let one = cluster_speakers(
            &embeddings,
            &durations,
            DiarizationOptions {
                similarity_threshold: 0.99,
                max_speakers: 1,
            },
        );
        assert!(one[..4].iter().all(|a| *a == one[0]));
    }

    #[test]
    fn tiny_clusters_are_absorbed() {
        let embeddings = vec![
            unit(&[1.0, 0.0, 0.0]),
            unit(&[1.0, 0.0, 0.0]),
            unit(&[0.0, 0.0, 1.0]), // a lone 1-second blip
        ];
        let durations = vec![100.0, 100.0, 1.0];
        let assignment = cluster_speakers(&embeddings, &durations, DiarizationOptions::default());
        assert_eq!(assignment[2], assignment[0]);
        // Short clip: a 4-second speaker out of 12 seconds is real, not noise.
        let short = vec![4.0, 4.0, 4.0];
        let kept = cluster_speakers(&embeddings, &short, DiarizationOptions::default());
        assert_ne!(kept[2], kept[0]);
    }

    #[test]
    fn single_track_labels_number_speakers_by_first_appearance() {
        let transcript = vec![
            t(0.0, 50.0, "hi"),
            t(50.0, 100.0, "hello"),
            t(100.0, 150.0, "again"),
        ];
        let embeddings = vec![unit(&[1.0, 0.0]), unit(&[0.0, 1.0]), unit(&[1.0, 0.0])];
        let labelled = label_single_track(&transcript, &embeddings, DiarizationOptions::default(), &[]);
        let names: Vec<&str> = labelled.iter().map(|l| l.speaker.as_str()).collect();
        assert_eq!(names, vec!["Speaker 1", "Speaker 2", "Speaker 1"]);
        assert_eq!(
            render_transcript(&labelled),
            "Speaker 1: hi\nSpeaker 2: hello\nSpeaker 1: again"
        );
        assert_eq!(speaker_names(&labelled), vec!["Speaker 1", "Speaker 2"]);
    }

    #[test]
    fn two_track_labels_owner_by_energy_and_others_from_clusters() {
        let transcript = vec![
            t(0.0, 50.0, "I think"),
            t(50.0, 100.0, "yes"),
            t(100.0, 150.0, "no"),
            t(150.0, 200.0, "ok"),
            t(200.0, 200.5, "mm"),
        ];
        let owner = vec![true, false, false, true, false];
        let embeddings = vec![None, unit(&[1.0, 0.0]), unit(&[0.0, 1.0]), None, None];
        let labelled = label_two_track(&transcript, &owner, &embeddings, DiarizationOptions::default(), "Giovanni", &[]);
        let names: Vec<&str> = labelled.iter().map(|l| l.speaker.as_str()).collect();
        assert_eq!(
            names,
            vec![
                "Giovanni",
                "Speaker 2",
                "Speaker 3",
                "Giovanni",
                "Speaker 3"
            ]
        );
        assert_eq!(
            render_transcript(&labelled),
            "Giovanni: I think\nSpeaker 2: yes\nSpeaker 3: no\nGiovanni: ok\nSpeaker 3: mm"
        );
    }

    #[test]
    fn known_voices_are_recognized_and_do_not_consume_numbers() {
        let transcript = vec![t(0.0, 50.0, "a"), t(50.0, 100.0, "b"), t(100.0, 150.0, "c")];
        let embeddings = vec![unit(&[1.0, 0.0, 0.0]), unit(&[0.0, 1.0, 0.0]), unit(&[0.0, 0.0, 1.0])];
        let known = vec![KnownSpeaker { name: "Giovanni".into(), is_owner: true, centroid: unit(&[0.95, 0.05, 0.0]).unwrap() }];
        let labelled = label_single_track(&transcript, &embeddings, DiarizationOptions::default(), &known);
        let names: Vec<&str> = labelled.iter().map(|l| l.speaker.as_str()).collect();
        assert_eq!(names, vec!["Giovanni", "Speaker 1", "Speaker 2"]);
        let stranger = vec![KnownSpeaker { name: "Nobody".into(), is_owner: false, centroid: unit(&[1.0, 1.0, 1.0]).unwrap() }]; // 0.58 to every axis, below the threshold
        let labelled = label_single_track(&transcript, &embeddings, DiarizationOptions::default(), &stranger);
        assert!(labelled.iter().all(|l| l.speaker.starts_with("Speaker")));
    }

    #[test]
    fn enrollment_windows_skip_silence() {
        let rate = 100;
        let mut clip = vec![0.2f32; 300];
        clip.extend(vec![0.0f32; 300]);
        clip.extend(vec![0.2f32; 200]);
        assert_eq!(enrollment_windows(&clip, rate), vec![(0, 300), (600, 800)]);
    }

    #[test]
    fn render_merges_consecutive_segments_and_skips_empty() {
        let segs = vec![
            LabelledSegment {
                start: 0.0,
                end: 1.0,
                speaker: "A".into(),
                text: "one".into(),
            },
            LabelledSegment {
                start: 1.0,
                end: 2.0,
                speaker: "A".into(),
                text: "  ".into(),
            },
            LabelledSegment {
                start: 2.0,
                end: 3.0,
                speaker: "A".into(),
                text: "two".into(),
            },
            LabelledSegment {
                start: 3.0,
                end: 4.0,
                speaker: "B".into(),
                text: "three".into(),
            },
        ];
        assert_eq!(render_transcript(&segs), "A: one two\nB: three");
    }

    #[test]
    fn owner_mask_follows_mic_energy() {
        let rate = 100;
        let mut left = vec![0.5f32; 100];
        left.extend(vec![0.05f32; 100]);
        left.extend(vec![0.0f32; 100]);
        let mut right = vec![0.05f32; 100];
        right.extend(vec![0.5f32; 100]);
        right.extend(vec![0.0f32; 100]);
        let transcript = vec![t(0.0, 1.0, ""), t(1.0, 2.0, ""), t(2.0, 3.0, "")];
        assert_eq!(
            owner_mask(&left, &right, rate, &transcript, OWNER_ENERGY_RATIO),
            vec![true, false, false]
        );
    }

    #[test]
    fn identical_tracks_are_detected() {
        let left: Vec<f32> = (0..1000).map(|i| ((i as f32) * 0.1).sin()).collect();
        let right = left.clone();
        assert!(tracks_correlation(&left, &right) > IDENTICAL_TRACKS_CORRELATION);
        let other: Vec<f32> = (0..1000).map(|i| ((i as f32) * 0.37 + 1.0).cos()).collect();
        assert!(tracks_correlation(&left, &other) < 0.5);
    }

    #[test]
    fn wav_channels_are_split() {
        let dir = std::env::temp_dir().join(format!("scriba-diar-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("stereo.wav");
        let spec = hound::WavSpec {
            channels: 2,
            sample_rate: 16_000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut w = hound::WavWriter::create(&path, spec).unwrap();
        for i in 0..4 {
            w.write_sample((i * 1000) as i16).unwrap();
            w.write_sample(-(i * 1000) as i16).unwrap();
        }
        w.finalize().unwrap();
        let (channels, rate) = read_wav_channels(&path).unwrap();
        assert_eq!(rate, 16_000);
        assert_eq!(channels.len(), 2);
        assert_eq!(channels[0].len(), 4);
        assert!(channels[0][3] > 0.09 && channels[1][3] < -0.09);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
